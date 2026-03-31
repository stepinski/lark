// cmd/fwfetch is a quick CLI to verify FlowWorks connectivity and pull
// the first 30 days of data from the two Peel sites.
//
// Usage:
//
//	go run ./cmd/fwfetch -user YOUR_USER -pass YOUR_PASS
//	go run ./cmd/fwfetch -user YOUR_USER -pass YOUR_PASS -days 365
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/stepinski/lark/datasource/flowworks"
)

func main() {
	user := flag.String("user", os.Getenv("FW_USER"), "FlowWorks username (or set FW_USER env)")
	pass := flag.String("pass", os.Getenv("FW_PASS"), "FlowWorks password (or set FW_PASS env)")
	days := flag.Int("days", 30, "Number of days to fetch")
	baseURL := flag.String("url", "https://developers.flowworks.com/fwapi/v2", "FlowWorks API base URL")
	flag.Parse()

	if *user == "" || *pass == "" {
		log.Fatal("credentials required: -user and -pass flags, or FW_USER / FW_PASS env vars")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	c := flowworks.NewClient(*baseURL, *user, *pass)

	// --- verify connectivity ---
	fmt.Println("→ authenticating...")
	sites, err := c.Sites(ctx)
	if err != nil {
		log.Fatalf("sites error: %v", err)
	}
	fmt.Printf("✓ connected — %d sites visible\n\n", len(sites))

	// --- known Peel sites ---
	type siteSpec struct {
		siteID    int
		siteName  string
		channels  map[string]int
	}

	peelSites := []siteSpec{
		{
			siteID:   241,
			siteName: "Cavendish Cr (OVF)",
			channels: map[string]int{
				"Depth":    36843,
				"Rainfall": 21881,
				"Float":    36451,
			},
		},
		{
			siteID:   255,
			siteName: "Clarkson GO Weir (OVF1)",
			channels: map[string]int{
				"Depth":    36930,
				"Rainfall": 36503,
				"Float":    36493,
			},
		},
	}

	opt := flowworks.LastN("D", *days)

	for _, site := range peelSites {
		fmt.Printf("=== Site %d: %s ===\n", site.siteID, site.siteName)

		channelIDs := make([]int, 0, len(site.channels))
		idToName := make(map[int]string)
		for name, id := range site.channels {
			channelIDs = append(channelIDs, id)
			idToName[id] = name
		}

		data, err := c.MultiChannelData(ctx, site.siteID, channelIDs, opt)
		if err != nil {
			fmt.Printf("  ✗ error: %v\n\n", err)
			continue
		}

		for id, pts := range data {
			name := idToName[id]
			stats := computeStats(pts)
			fmt.Printf("  %-10s (ch %d): %d points  min=%.3f  max=%.3f  mean=%.3f  nan=%d\n",
				name, id, len(pts), stats.min, stats.max, stats.mean, stats.nanCount)
		}
		fmt.Println()
	}
}

type stats struct {
	min, max, mean float64
	nanCount       int
}

func computeStats(pts []flowworks.DataPoint) stats {
	if len(pts) == 0 {
		return stats{}
	}
	s := stats{min: math.Inf(1), max: math.Inf(-1)}
	var sum float64
	var count int
	for _, p := range pts {
		if math.IsNaN(p.Value) {
			s.nanCount++
			continue
		}
		if p.Value < s.min {
			s.min = p.Value
		}
		if p.Value > s.max {
			s.max = p.Value
		}
		sum += p.Value
		count++
	}
	if count > 0 {
		s.mean = sum / float64(count)
	}
	return s
}
