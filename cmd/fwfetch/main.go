// cmd/fwfetch pulls time series data from a FlowWorks API endpoint.
//
// Usage:
//
//	go run ./cmd/fwfetch -list-sites
//	go run ./cmd/fwfetch -site 241 -list-channels
//	go run ./cmd/fwfetch -site 241 -channels 10,11,12 -days 30
//	go run ./cmd/fwfetch -site 241 -channels 10,11 -start 2024-01-01 -end 2026-01-01 -csv output.csv
//
// Credentials via environment variables:
//
//	export FW_USER=your_username
//	export FW_PASS=your_password
package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/stepinski/lark/datasource/flowworks"
)

func main() {
	user        := flag.String("user", os.Getenv("FW_USER"), "FlowWorks username (or FW_USER env)")
	pass        := flag.String("pass", os.Getenv("FW_PASS"), "FlowWorks password (or FW_PASS env)")
	baseURL     := flag.String("url", "https://developers.flowworks.com/fwapi/v2", "FlowWorks API base URL")
	siteID      := flag.Int("site", 0, "Site ID")
	channelList := flag.String("channels", "", "Comma-separated channel IDs, e.g. 10,11,12")
	days        := flag.Int("days", 30, "Number of recent days to fetch")
	start       := flag.String("start", "", "Start date: yyyy-MM-dd")
	end         := flag.String("end", "", "End date: yyyy-MM-dd")
	csvOut      := flag.String("csv", "", "Write output to CSV file (e.g. output.csv)")
	listSites   := flag.Bool("list-sites", false, "List all visible sites and exit")
	listChans   := flag.Bool("list-channels", false, "List all channels for -site and exit")
	flag.Parse()

	if *user == "" || *pass == "" {
		log.Fatal("credentials required: -user / -pass flags or FW_USER / FW_PASS env vars")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	c := flowworks.NewClient(*baseURL, *user, *pass)

	// --- list sites ---
	if *listSites {
		sites, err := c.Sites(ctx)
		if err != nil {
			log.Fatalf("sites error: %v", err)
		}
		fmt.Printf("%-8s %-40s %s\n", "ID", "Name", "Types")
		fmt.Println(strings.Repeat("-", 70))
		for _, s := range sites {
			fmt.Printf("%-8d %-40s %s\n", s.SiteID, s.SiteName, strings.Join(s.SiteTypes, ","))
		}
		return
	}

	// --- list channels ---
	if *listChans {
		if *siteID == 0 {
			log.Fatal("-site required with -list-channels")
		}
		channels, err := c.SiteChannels(ctx, *siteID)
		if err != nil {
			log.Fatalf("channels error: %v", err)
		}
		fmt.Printf("%-8s %-40s %-8s %s\n", "ID", "Name", "Units", "Type")
		fmt.Println(strings.Repeat("-", 70))
		for _, ch := range channels {
			fmt.Printf("%-8d %-40s %-8s %s\n", ch.ChannelID, ch.ChannelName, ch.Units, ch.ChannelType)
		}
		return
	}

	// --- fetch data ---
	if *siteID == 0 {
		log.Fatal("-site required")
	}
	if *channelList == "" {
		log.Fatal("-channels required, e.g. -channels 10,11,12")
	}

	channelIDs, err := parseChannelIDs(*channelList)
	if err != nil {
		log.Fatalf("invalid -channels: %v", err)
	}

	var opt flowworks.QueryOption
	if *start != "" {
		opt = flowworks.DateRange(*start, *end)
	} else {
		opt = flowworks.LastN("D", *days)
	}

	fmt.Fprintf(os.Stderr, "→ fetching site %d, channels %v\n", *siteID, channelIDs)

	data, err := c.MultiChannelData(ctx, *siteID, channelIDs, opt)
	if err != nil {
		log.Fatalf("fetch error: %v", err)
	}

	// --- print summary to stderr always ---
	fmt.Fprintf(os.Stderr, "\n%-12s %-8s %-10s %-10s %-10s %-10s %s\n",
		"Channel", "Points", "Min", "Max", "Mean", "NaN", "First timestamp")
	fmt.Fprintln(os.Stderr, strings.Repeat("-", 75))
	for _, id := range channelIDs {
		pts := data[id]
		s := computeStats(pts)
		first := ""
		if len(pts) > 0 {
			first = pts[0].Time.Format("2006-01-02 15:04")
		}
		fmt.Fprintf(os.Stderr, "%-12d %-8d %-10.3f %-10.3f %-10.3f %-10d %s\n",
			id, len(pts), s.min, s.max, s.mean, s.nanCount, first)
	}

	// --- CSV output ---
	if *csvOut != "" {
		if err := writeCSV(*csvOut, channelIDs, data); err != nil {
			log.Fatalf("csv write error: %v", err)
		}
		fmt.Fprintf(os.Stderr, "\n✓ wrote %s\n", *csvOut)
	}
}

// writeCSV writes all channels into a single CSV with columns:
// timestamp, channel_<id1>, channel_<id2>, ...
// Timestamps are aligned — missing values for a channel at a given
// timestamp are written as empty string.
func writeCSV(path string, channelIDs []int, data map[int][]flowworks.DataPoint) error {
	// build unified sorted timestamp index
	tsSet := make(map[int64]struct{})
	for _, pts := range data {
		for _, p := range pts {
			tsSet[p.Time.Unix()] = struct{}{}
		}
	}
	timestamps := make([]int64, 0, len(tsSet))
	for ts := range tsSet {
		timestamps = append(timestamps, ts)
	}
	// sort timestamps
	for i := 1; i < len(timestamps); i++ {
		for j := i; j > 0 && timestamps[j] < timestamps[j-1]; j-- {
			timestamps[j], timestamps[j-1] = timestamps[j-1], timestamps[j]
		}
	}

	// build lookup: channelID -> timestamp -> value
	lookup := make(map[int]map[int64]string, len(channelIDs))
	for _, id := range channelIDs {
		lookup[id] = make(map[int64]string, len(data[id]))
		for _, p := range data[id] {
			if math.IsNaN(p.Value) {
				lookup[id][p.Time.Unix()] = ""
			} else {
				lookup[id][p.Time.Unix()] = strconv.FormatFloat(p.Value, 'f', 4, 64)
			}
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)

	// header
	header := []string{"timestamp"}
	for _, id := range channelIDs {
		header = append(header, fmt.Sprintf("ch_%d", id))
	}
	if err := w.Write(header); err != nil {
		return err
	}

	// rows
	for _, ts := range timestamps {
		row := []string{time.Unix(ts, 0).UTC().Format("2006-01-02T15:04:05Z")}
		for _, id := range channelIDs {
			v := lookup[id][ts]
			row = append(row, v)
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	w.Flush()
	return w.Error()
}

func parseChannelIDs(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("%q is not a valid channel ID", p)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("no valid channel IDs in %q", s)
	}
	return ids, nil
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
