// Package flowworks provides a typed Go client for the FlowWorks Web API v2.
//
// The client handles:
//   - JWT authentication with automatic token refresh (tokens expire after 60min)
//   - Paginated data fetching respecting the ~1.5M point response cap
//   - Typed structs that map directly to Lark's internal primitives
//   - Context-aware requests for graceful cancellation
//
// Usage:
//
//	c, err := flowworks.NewClient("https://developers.flowworks.com/fwapi/v2", "user", "pass")
//	pts, err := c.ChannelData(ctx, 241, 36843,
//	    flowworks.DateRange("2022-01-01", "2024-01-01"))
package flowworks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"
)

const (
	// tokenRefreshMargin is how early we refresh before expiry.
	tokenRefreshMargin = 5 * time.Minute
	// defaultTimeout for individual HTTP requests.
	defaultTimeout = 30 * time.Second
	// paginationChunkDays is the chunk size used when auto-paginating.
	// 90 days at 5-min resolution = 25,920 points — well under the cap.
	paginationChunkDays = 90
)

// --- API response envelope ---

// apiResponse is the generic FlowWorks envelope.
type apiResponse[T any] struct {
	Resources     []T    `json:"Resources"`
	ResultCode    int    `json:"ResultCode"`
	ResultMessage string `json:"ResultMessage"`
}

// --- Auth types ---

type tokenRequest struct {
	UserName string `json:"Username"`
	Password string `json:"Password"`
}

type tokenResponse struct {
	Token    string `json:"Token"`
	IssuedAt string `json:"IssuedAt"`
	Expires  string `json:"Expires"`
}

// --- Domain types ---

// DataPoint is a single timestamped sensor reading from the FlowWorks API.
// DataValue is a string in the API response — we parse it to float64.
type DataPoint struct {
	// Time is the UTC observation time.
	Time time.Time
	// Value is the parsed numeric reading. NaN if the raw value was non-numeric.
	Value float64
	// Raw is the original string value as returned by the API.
	Raw string
}

// Site is a FlowWorks monitoring site.
type Site struct {
	SiteID       int      `json:"Id"`
	SiteName     string   `json:"Name"`
	InternalName string   `json:"InternalName"`
	Longitude    string   `json:"Longitude"`
	Latitude     string   `json:"Latitude"`
	SiteTypes    []string `json:"SiteTypes"`
}

// Channel is a data channel within a site.
type Channel struct {
	ChannelID         int    `json:"Id"`
	ChannelName       string `json:"Name"`
	Units             string `json:"Unit"`
	ChannelType       string `json:"ChannelType"`
	IsVisible         bool   `json:"IsVisible"`
	IsRainfallEnabled bool   `json:"IsRainfallEnabled"`
}

// --- Query options ---

// QueryOption configures a data fetch request.
type QueryOption func(*queryParams)

type queryParams struct {
	startDate string
	endDate   string
	interval  string
	number    string
}

// DateRange fetches data between start and end (format: "yyyy-MM-dd" or
// "yyyy-MM-ddTHH:mm:ss").
func DateRange(start, end string) QueryOption {
	return func(p *queryParams) {
		p.startDate = start
		p.endDate = end
	}
}

// LastN fetches the most recent N intervals.
// intervalType: "Y", "M", "D", "HH", "MM", "SS"
// Example: LastN("D", 30) → last 30 days.
func LastN(intervalType string, n int) QueryOption {
	return func(p *queryParams) {
		p.interval = intervalType
		p.number = strconv.Itoa(n)
	}
}

// --- Client ---

// Client is the FlowWorks API client. Safe for concurrent use.
type Client struct {
	baseURL    string
	username   string
	password   string
	httpClient *http.Client

	mu          sync.Mutex
	token       string
	tokenExpiry time.Time
}

// NewClient creates an authenticated FlowWorks client.
// It does not immediately authenticate — the first request triggers auth.
func NewClient(baseURL, username, password string) *Client {
	return &Client{
		baseURL:  baseURL,
		username: username,
		password: password,
		httpClient: &http.Client{
			Timeout: defaultTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 10 {
					return fmt.Errorf("flowworks: too many redirects")
				}
				return nil
			},
		},
	}
}

// authenticate fetches a new JWT token. Called automatically when the token
// is absent or nearing expiry.
func (c *Client) authenticate(ctx context.Context) error {
	body, err := json.Marshal(tokenRequest{
		UserName: c.username,
		Password: c.password,
	})
	if err != nil {
		return fmt.Errorf("flowworks: marshal auth request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/authenticate", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("flowworks: create auth request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("flowworks: auth request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("flowworks: auth HTTP %d", resp.StatusCode)
	}

	var tr tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return fmt.Errorf("flowworks: decode auth response: %w", err)
	}

	expiry := parseExpiry(tr.Expires)

	c.mu.Lock()
	c.token = tr.Token
	c.tokenExpiry = expiry
	c.mu.Unlock()

	return nil
}

// token returns a valid JWT, refreshing if necessary.
func (c *Client) ensureToken(ctx context.Context) (string, error) {
	c.mu.Lock()
	needsRefresh := c.token == "" || time.Now().UTC().After(c.tokenExpiry.Add(-tokenRefreshMargin))
	c.mu.Unlock()

	if needsRefresh {
		if err := c.authenticate(ctx); err != nil {
			return "", err
		}
	}

	c.mu.Lock()
	t := c.token
	c.mu.Unlock()
	return t, nil
}

// get performs an authenticated GET request and decodes the response.
func (c *Client) get(ctx context.Context, path string, dest interface{}) error {
	token, err := c.ensureToken(ctx)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("flowworks: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("flowworks: GET %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		// token may have just expired — force refresh and retry once
		if err := c.authenticate(ctx); err != nil {
			return fmt.Errorf("flowworks: re-auth after 401: %w", err)
		}
		return c.get(ctx, path, dest)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("flowworks: GET %s HTTP %d: %s", path, resp.StatusCode, body)
	}

	return json.NewDecoder(resp.Body).Decode(dest)
}

// --- Public API ---

// Sites returns all sites visible to the authenticated user.
func (c *Client) Sites(ctx context.Context) ([]Site, error) {
	var resp apiResponse[Site]
	if err := c.get(ctx, "/sites", &resp); err != nil {
		return nil, err
	}
	if resp.ResultCode != 0 {
		return nil, fmt.Errorf("flowworks: Sites API error %d: %s", resp.ResultCode, resp.ResultMessage)
	}
	return resp.Resources, nil
}

// SiteChannels returns all channels for the given site.
func (c *Client) SiteChannels(ctx context.Context, siteID int) ([]Channel, error) {
	var resp apiResponse[Channel]
	path := fmt.Sprintf("/sites/%d/channels", siteID)
	if err := c.get(ctx, path, &resp); err != nil {
		return nil, err
	}
	if resp.ResultCode != 0 {
		return nil, fmt.Errorf("flowworks: SiteChannels API error %d: %s", resp.ResultCode, resp.ResultMessage)
	}
	return resp.Resources, nil
}

// rawDataPoint is the wire format from the API.
// DataTime is returned without timezone (e.g. "2026-03-01T06:35:00")
// so we unmarshal it as a string and parse manually.
type rawDataPoint struct {
	DataValue string `json:"DataValue"`
	DataTime  string `json:"DataTime"`
}

// parsedTime parses the FlowWorks timestamp which has no timezone suffix.
// Treated as UTC.
func parsedTime(s string) time.Time {
	for _, layout := range []string{
		"2006-01-02T15:04:05",
		"2006-01-02T15:04:05Z",
		time.RFC3339,
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

// ChannelData fetches data points for a single channel.
// Automatically paginates in 90-day chunks to stay under the 1.5M point cap.
// Returns DataPoints in chronological order.
func (c *Client) ChannelData(ctx context.Context, siteID, channelID int, opts ...QueryOption) ([]DataPoint, error) {
	p := &queryParams{}
	for _, o := range opts {
		o(p)
	}

	// if date range provided, paginate in chunks
	if p.startDate != "" {
		return c.paginatedFetch(ctx, siteID, channelID, p)
	}

	// interval-based fetch — single request
	return c.fetchChunk(ctx, siteID, channelID, p)
}

// paginatedFetch splits a date range into 90-day chunks and merges results.
func (c *Client) paginatedFetch(ctx context.Context, siteID, channelID int, p *queryParams) ([]DataPoint, error) {
	start, err := parseFlexDate(p.startDate)
	if err != nil {
		return nil, fmt.Errorf("flowworks: parse startDate %q: %w", p.startDate, err)
	}

	end := time.Now().UTC()
	if p.endDate != "" {
		end, err = parseFlexDate(p.endDate)
		if err != nil {
			return nil, fmt.Errorf("flowworks: parse endDate %q: %w", p.endDate, err)
		}
	}

	var all []DataPoint
	chunkStart := start

	for chunkStart.Before(end) {
		chunkEnd := chunkStart.AddDate(0, 0, paginationChunkDays)
		if chunkEnd.After(end) {
			chunkEnd = end
		}

		chunk := &queryParams{
			startDate: chunkStart.Format("2006-01-02T15:04:05"),
			endDate:   chunkEnd.Format("2006-01-02T15:04:05"),
		}

		pts, err := c.fetchChunk(ctx, siteID, channelID, chunk)
		if err != nil {
			return nil, fmt.Errorf("flowworks: chunk %s–%s: %w",
				chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), err)
		}
		all = append(all, pts...)

		// check context between chunks
		select {
		case <-ctx.Done():
			return all, ctx.Err()
		default:
		}

		chunkStart = chunkEnd
	}

	return all, nil
}

// fetchChunk performs a single data request for the given params.
func (c *Client) fetchChunk(ctx context.Context, siteID, channelID int, p *queryParams) ([]DataPoint, error) {
	q := url.Values{}
	if p.startDate != "" {
		q.Set("startDateFilter", p.startDate)
	}
	if p.endDate != "" {
		q.Set("endDateFilter", p.endDate)
	}
	if p.interval != "" {
		q.Set("intervalTypeFilter", p.interval)
		q.Set("intervalNumberFilter", p.number)
	}

	path := fmt.Sprintf("/sites/%d/channels/%d/data?%s", siteID, channelID, q.Encode())

	var resp apiResponse[rawDataPoint]
	if err := c.get(ctx, path, &resp); err != nil {
		return nil, err
	}
	if resp.ResultCode != 0 {
		return nil, fmt.Errorf("flowworks: data API error %d: %s", resp.ResultCode, resp.ResultMessage)
	}

	pts := make([]DataPoint, 0, len(resp.Resources))
	for _, r := range resp.Resources {
		v, err := strconv.ParseFloat(r.DataValue, 64)
		dp := DataPoint{
			Time: parsedTime(r.DataTime),
			Raw:  r.DataValue,
		}
		if err == nil {
			dp.Value = v
		} else {
			dp.Value = math.NaN()
		}
		pts = append(pts, dp)
	}

	return pts, nil
}

// MultiChannelData fetches multiple channels for a site concurrently and
// returns a map of channelID → []DataPoint. All channels use the same
// query options. Errors from individual channels are collected and returned
// as a MultiError.
func (c *Client) MultiChannelData(ctx context.Context, siteID int, channelIDs []int, opts ...QueryOption) (map[int][]DataPoint, error) {
	type result struct {
		id  int
		pts []DataPoint
		err error
	}

	ch := make(chan result, len(channelIDs))
	for _, id := range channelIDs {
		go func(cid int) {
			pts, err := c.ChannelData(ctx, siteID, cid, opts...)
			ch <- result{id: cid, pts: pts, err: err}
		}(id)
	}

	out := make(map[int][]DataPoint, len(channelIDs))
	var errs []error
	for range channelIDs {
		r := <-ch
		if r.err != nil {
			errs = append(errs, fmt.Errorf("channel %d: %w", r.id, r.err))
		} else {
			out[r.id] = r.pts
		}
	}

	if len(errs) > 0 {
		return out, &MultiError{Errors: errs}
	}
	return out, nil
}

// --- helpers ---

// parseFlexDate parses "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm:ss".
func parseFlexDate(s string) (time.Time, error) {
	for _, layout := range []string{
		"2006-01-02T15:04:05",
		"2006-01-02",
	} {
		if t, err := time.ParseInLocation(layout, s, time.UTC); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised date format %q", s)
}

// parseExpiry parses the FlowWorks token expiry string.
// The API returns "2006-01-02 15:04:05.000Z" (with Z suffix).
// Falls back to 60-minute validity if parsing fails.
func parseExpiry(s string) time.Time {
	for _, layout := range []string{
		"2006-01-02 15:04:05.000Z",
		"2006-01-02 15:04:05.000",
		"2006-01-02 15:04:05Z",
		"2006-01-02 15:04:05",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC()
		}
	}
	return time.Now().UTC().Add(60 * time.Minute)
}

// MultiError aggregates multiple channel fetch errors.
type MultiError struct {
	Errors []error
}

func (e *MultiError) Error() string {
	msg := fmt.Sprintf("flowworks: %d channel error(s):", len(e.Errors))
	for _, err := range e.Errors {
		msg += "\n  - " + err.Error()
	}
	return msg
}
