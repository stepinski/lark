// Package pipeline provides shared observation types, feature engineering,
// and site discovery helpers used by cmd/evaluate and cmd/serve.
package pipeline

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/stepinski/lark/datasource/flowworks"
	"github.com/stepinski/lark/models/dwf"
)

// Obs is a single aligned observation with depth, rain, and optional upstream data.
type Obs struct {
	T          time.Time
	DepthMM    float64
	Rain6hr    float64
	RainGARR   float64
	UpstreamMM float64
	FloatMM    float64 // overflow pipe float sensor (0=normal, >0.5=overflow)
}

// FindChannel returns the best-matching channel for the given keywords.
// It prefers exact matches over substring matches to avoid false positives
// (e.g. "Depth" should match "Depth", not "AV Depth Surcharge Indicator").
func FindChannel(channels []flowworks.Channel, keywords []string) *int {
	// First pass: exact match
	for _, ch := range channels {
		for _, kw := range keywords {
			if ch.ChannelName == kw {
				id := ch.ChannelID
				return &id
			}
		}
	}
	// Second pass: substring match (fallback)
	for _, ch := range channels {
		for _, kw := range keywords {
			if strings.Contains(ch.ChannelName, kw) {
				id := ch.ChannelID
				return &id
			}
		}
	}
	return nil
}

// FetchInvert retrieves the current invert level from the overflow pipe channel.
func FetchInvert(ctx context.Context, client *flowworks.Client, siteID, channelID int) (float64, error) {
	data, err := client.ChannelData(ctx, siteID, channelID, flowworks.LastN("D", 7))
	if err != nil {
		return 0, err
	}
	if len(data) == 0 {
		return 0, fmt.Errorf("no data for overflow pipe channel")
	}
	return data[0].Value, nil
}

// BuildDataset constructs feature vectors and labels from aligned observations.
// Features: excess (depth - invert), AMC (DWF API), rain1hr, month sin, month cos.
// Label: 1.0 if depth >= invert within next horizon steps, 0.0 otherwise.
func BuildDataset(rows []Obs, invert float64, horizon int, useGARR, useUpstream bool) ([][]float64, []float64, []time.Time) {
	var X [][]float64
	var y []float64
	var ts []time.Time

	for i := 288; i < len(rows)-horizon; i++ {
		excess := rows[i].DepthMM - invert
		amc := dwf.API(RecentRain(rows, i, 2016), 0.85)
		rain1hr := RollingSum(rows[:i+1], 12)
		mSin, mCos := MonthEncoding(rows[i].T)

		row := []float64{excess, amc, rain1hr, mSin, mCos}

		label := 0.0
		for j := 0; j < horizon; j++ {
			// Use float channel if available; fallback to depth >= invert
			if rows[i+j].FloatMM >= 0.5 {
				label = 1.0
				break
			}
			if rows[i+j].DepthMM >= invert {
				label = 1.0
				break
			}
		}

		X = append(X, row)
		y = append(y, label)
		ts = append(ts, rows[i].T)
	}

	return X, y, ts
}

// RecentRain returns the rain values from the last n observations up to index idx.
func RecentRain(rows []Obs, idx, n int) []float64 {
	start := idx - n + 1
	if start < 0 {
		start = 0
	}
	rain := make([]float64, idx-start+1)
	for i, r := range rows[start : idx+1] {
		rain[i] = r.Rain6hr
	}
	return rain
}

// RollingSum returns the sum of the last n rain values.
func RollingSum(rows []Obs, n int) float64 {
	var sum float64
	for i := len(rows) - 1; i >= 0 && i >= len(rows)-n; i-- {
		sum += rows[i].Rain6hr
	}
	return sum
}

// MonthEncoding returns sin/cos encoding of the month for cyclical features.
func MonthEncoding(t time.Time) (float64, float64) {
	m := float64(t.Month()) / 12.0
	return math.Sin(2 * math.Pi * m), math.Cos(2 * math.Pi * m)
}

// AlignChannels merges multi-channel data into a time-sorted slice of Obs.
func AlignChannels(data map[int][]flowworks.DataPoint, depthID, rainID int) []Obs {
	tsMap := make(map[string]map[int]float64)
	for chID, pts := range data {
		for _, p := range pts {
			key := p.Time.Format(time.RFC3339)
			if tsMap[key] == nil {
				tsMap[key] = make(map[int]float64)
			}
			tsMap[key][chID] = p.Value
		}
	}

	var timestamps []string
	for ts := range tsMap {
		timestamps = append(timestamps, ts)
	}
	sort.Strings(timestamps)

	var rows []Obs
	for _, ts := range timestamps {
		t, _ := time.Parse(time.RFC3339, ts)
		rows = append(rows, Obs{
			T:       t,
			DepthMM: tsMap[ts][depthID],
			Rain6hr: tsMap[ts][rainID],
		})
	}
	return rows
}

// DiscoverOVFSites filters FlowWorks sites to OVF (overflow) sites, excluding SPS.
func DiscoverOVFSites(ctx context.Context, client *flowworks.Client) ([]flowworks.Site, error) {
	sites, err := client.Sites(ctx)
	if err != nil {
		return nil, fmt.Errorf("list sites: %w", err)
	}

	var ovfSites []flowworks.Site
	for _, s := range sites {
		if strings.Contains(s.SiteName, "OVF") && !strings.Contains(s.SiteName, "SPS") {
			ovfSites = append(ovfSites, s)
		}
	}
	return ovfSites, nil
}
