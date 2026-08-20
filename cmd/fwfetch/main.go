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
	user := flag.String("user", os.Getenv("FW_USER"), "FlowWorks username (or FW_USER env)")
	pass := flag.String("pass", os.Getenv("FW_PASS"), "FlowWorks password (or FW_PASS env)")
	baseURL := flag.String("url", "https://developers.flowworks.com/fwapi/v2", "FlowWorks API base URL")
	siteID := flag.Int("site", 0, "Site ID")
	days := flag.Int("days", 30, "Number of recent days to fetch")
	start := flag.String("start", "", "Start date: yyyy-MM-dd")
	end := flag.String("end", "", "End date: yyyy-MM-dd")
	csvOut := flag.String("csv", "", "Write output to CSV file")
	listSites := flag.Bool("list-sites", false, "List all visible sites and exit")
	listChans := flag.Bool("list-channels", false, "List all channels for -site and exit")
	channels := flag.String("channels", "", "Comma-separated channel IDs to fetch")
	flag.Parse()

	if *user == "" || *pass == "" {
		log.Fatal("FW_USER and FW_PASS required (flags or env vars)")
	}

	ctx := context.Background()
	client := flowworks.NewClient(*baseURL, *user, *pass)

	if *listSites {
		sites, err := client.Sites(ctx)
		if err != nil {
			log.Fatalf("list sites: %v", err)
		}
		fmt.Printf("%-10s %-40s %s\n", "SiteID", "Name", "Type")
		for _, s := range sites {
			fmt.Printf("%-10d %-40s %s\n", s.SiteID, s.SiteName, s.SiteType)
		}
		return
	}

	if *siteID == 0 {
		log.Fatal("-site required")
	}

	if *listChans {
		chans, err := client.SiteChannels(ctx, *siteID)
		if err != nil {
			log.Fatalf("list channels: %v", err)
		}
		fmt.Printf("%-12s %-40s %s\n", "ChannelID", "Name", "Units")
		for _, ch := range chans {
			fmt.Printf("%-12d %-40s %s\n", ch.ChannelID, ch.ChannelName, ch.Units)
		}
		return
	}

	if *channels == "" {
		log.Fatal("-channels required (comma-separated channel IDs)")
	}

	var chanIDs []int
	for _, s := range strings.Split(*channels, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		id, err := strconv.Atoi(s)
		if err != nil {
			log.Fatalf("invalid channel ID %q: %v", s, err)
		}
		chanIDs = append(chanIDs, id)
	}

	var opt flowworks.QueryOption
	if *start != "" {
		endDate := *end
		if endDate == "" {
			endDate = time.Now().UTC().Format("2006-01-02")
		}
		opt = flowworks.DateRange(*start, endDate)
		fmt.Printf("Fetching site %d channels %v from %s to %s...\n", *siteID, chanIDs, *start, endDate)
	} else {
		opt = flowworks.LastN("D", *days)
		fmt.Printf("Fetching site %d channels %v last %d days...\n", *siteID, chanIDs, *days)
	}

	data, err := client.MultiChannelData(ctx, *siteID, chanIDs, opt)
	if err != nil {
		log.Fatalf("fetch: %v", err)
	}

	for cid, pts := range data {
		fmt.Printf("  channel %d: %d points\n", cid, len(pts))
	}

	if *csvOut != "" {
		if err := writeCSV(*csvOut, chanIDs, data); err != nil {
			log.Fatalf("write CSV: %v", err)
		}
		fmt.Printf("Written to %s\n", *csvOut)
	}
}

func writeCSV(path string, chanIDs []int, data map[int][]flowworks.DataPoint) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)

	header := []string{"timestamp"}
	for _, id := range chanIDs {
		header = append(header, fmt.Sprintf("ch_%d", id))
	}
	_ = w.Write(header)

	if len(chanIDs) == 0 {
		w.Flush()
		return w.Error()
	}

	primary := data[chanIDs[0]]
	for i, pt := range primary {
		row := []string{pt.Time.Format(time.RFC3339)}
		for _, id := range chanIDs {
			pts := data[id]
			if i < len(pts) {
				v := pts[i].Value
				if math.IsNaN(v) {
					row = append(row, "")
				} else {
					row = append(row, fmt.Sprintf("%.4f", v))
				}
			} else {
				row = append(row, "")
			}
		}
		_ = w.Write(row)
	}

	w.Flush()
	return w.Error()
}
