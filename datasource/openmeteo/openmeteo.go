// Package openmeteo provides a client for the Open-Meteo weather forecast API.
//
// Open-Meteo is a free, open-source weather API with no authentication
// required. It provides hourly precipitation forecasts globally with up to
// 16-day horizon.
//
// API reference: https://open-meteo.com/en/docs
//
// Usage:
//
//	c := openmeteo.NewClient()
//	forecast, err := c.HourlyForecast(ctx, 43.55, -79.65, 2)
//	// returns hourly precipitation in mm for next 2 days
//
//	// for validation: what did the model predict at a past time?
//	hindcast, err := c.HindcastForecast(ctx, 43.55, -79.65,
//	    time.Date(2025, 4, 2, 18, 0, 0, 0, time.UTC), 12)
package openmeteo

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

const baseURL = "https://api.open-meteo.com/v1"

// HourlyPoint is a single hourly precipitation reading.
type HourlyPoint struct {
	// T is the timestamp in UTC.
	T time.Time
	// PrecipMM is precipitation in mm for that hour.
	PrecipMM float64
}

// Client is the Open-Meteo API client. Safe for concurrent use.
type Client struct {
	http    *http.Client
	baseURL string
}

// NewClient creates a new Open-Meteo client with sensible defaults.
func NewClient() *Client {
	return &Client{
		http:    &http.Client{Timeout: 30 * time.Second},
		baseURL: baseURL,
	}
}

// HourlyForecast fetches hourly precipitation forecast for the given
// coordinates over the next forecastDays days (max 16).
// Returns points in chronological order.
func (c *Client) HourlyForecast(ctx context.Context, lat, lon float64, forecastDays int) ([]HourlyPoint, error) {
	if forecastDays < 1 || forecastDays > 16 {
		return nil, fmt.Errorf("openmeteo: forecastDays must be 1-16, got %d", forecastDays)
	}

	q := url.Values{}
	q.Set("latitude", fmt.Sprintf("%.4f", lat))
	q.Set("longitude", fmt.Sprintf("%.4f", lon))
	q.Set("hourly", "precipitation")
	q.Set("forecast_days", fmt.Sprintf("%d", forecastDays))
	q.Set("timezone", "UTC")
	q.Set("precipitation_unit", "mm")

	endpoint := c.baseURL + "/forecast?" + q.Encode()
	return c.fetch(ctx, endpoint)
}

// HindcastForecast fetches what Open-Meteo's model predicted at a specific
// past time (startTime) for the following horizonHours hours.
//
// This uses the Open-Meteo historical forecast API which stores archive of
// past model runs. Useful for validating predictions against known events.
//
// Note: historical forecast data is available from 2022-01-01 onwards.
func (c *Client) HindcastForecast(ctx context.Context, lat, lon float64, startTime time.Time, horizonHours int) ([]HourlyPoint, error) {
	if horizonHours < 1 {
		return nil, fmt.Errorf("openmeteo: horizonHours must be >= 1")
	}

	endTime := startTime.Add(time.Duration(horizonHours) * time.Hour)

	q := url.Values{}
	q.Set("latitude", fmt.Sprintf("%.4f", lat))
	q.Set("longitude", fmt.Sprintf("%.4f", lon))
	q.Set("hourly", "precipitation")
	q.Set("start_date", startTime.UTC().Format("2006-01-02"))
	q.Set("end_date", endTime.UTC().Format("2006-01-02"))
	q.Set("timezone", "UTC")
	q.Set("precipitation_unit", "mm")
	q.Set("models", "best_match")

	// historical forecast API endpoint
	endpoint := "https://historical-forecast-api.open-meteo.com/v1/forecast?" + q.Encode()

	pts, err := c.fetch(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	// trim to requested window
	var out []HourlyPoint
	for _, p := range pts {
		if !p.T.Before(startTime) && !p.T.After(endTime) {
			out = append(out, p)
		}
	}
	return out, nil
}

// apiResponse is the Open-Meteo JSON response shape.
type apiResponse struct {
	Hourly struct {
		Time          []string  `json:"time"`
		Precipitation []float64 `json:"precipitation"`
	} `json:"hourly"`
	Error  bool   `json:"error"`
	Reason string `json:"reason"`
}

func (c *Client) fetch(ctx context.Context, endpoint string) ([]HourlyPoint, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("openmeteo: create request: %w", err)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openmeteo: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openmeteo: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openmeteo: HTTP %d: %s", resp.StatusCode, body)
	}

	var ar apiResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, fmt.Errorf("openmeteo: decode response: %w", err)
	}
	if ar.Error {
		return nil, fmt.Errorf("openmeteo: API error: %s", ar.Reason)
	}

	if len(ar.Hourly.Time) != len(ar.Hourly.Precipitation) {
		return nil, fmt.Errorf("openmeteo: time/precipitation length mismatch")
	}

	pts := make([]HourlyPoint, 0, len(ar.Hourly.Time))
	for i, ts := range ar.Hourly.Time {
		t, err := time.Parse("2006-01-02T15:04", ts)
		if err != nil {
			continue
		}
		pts = append(pts, HourlyPoint{
			T:        t.UTC(),
			PrecipMM: ar.Hourly.Precipitation[i],
		})
	}
	return pts, nil
}

// ResampleTo5Min resamples hourly precipitation to 5-minute intervals by
// distributing each hour's total evenly across 12 steps.
// This matches the FlowWorks sensor resolution.
func ResampleTo5Min(hourly []HourlyPoint) []HourlyPoint {
	out := make([]HourlyPoint, 0, len(hourly)*12)
	for _, h := range hourly {
		mmPer5min := h.PrecipMM / 12.0
		for step := 0; step < 12; step++ {
			out = append(out, HourlyPoint{
				T:        h.T.Add(time.Duration(step*5) * time.Minute),
				PrecipMM: mmPer5min,
			})
		}
	}
	return out
}

// Scenario represents a synthetic rain scenario for probabilistic forecasting.
type Scenario struct {
	Name     string
	MMPerHr  float64
}

// StandardScenarios returns the four standard rain scenarios for sewer
// overflow risk assessment.
func StandardScenarios() []Scenario {
	return []Scenario{
		{Name: "none",     MMPerHr: 0.0},
		{Name: "light",    MMPerHr: 0.5},
		{Name: "moderate", MMPerHr: 2.0},
		{Name: "heavy",    MMPerHr: 5.0},
	}
}

// SyntheticForecast generates a constant-rate rain scenario as 5-min points
// over horizonSteps steps starting from startTime.
func SyntheticForecast(startTime time.Time, horizonSteps int, mmPerHr float64) []HourlyPoint {
	mmPer5min := mmPerHr / 12.0
	out := make([]HourlyPoint, horizonSteps)
	for i := range out {
		out[i] = HourlyPoint{
			T:        startTime.Add(time.Duration(i*5) * time.Minute),
			PrecipMM: mmPer5min,
		}
	}
	return out
}
