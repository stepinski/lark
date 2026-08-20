package main

import (
	"encoding/json"
	"testing"

	"github.com/stepinski/lark/datasource/flowworks"
)

func TestSitesToJSON(t *testing.T) {
	sites := []flowworks.Site{
		{SiteID: 1, SiteName: "Cavendish Creek", SiteType: "depth"},
		{SiteID: 2, SiteName: "Main Street", SiteType: "rainfall"},
	}

	raw, err := json.Marshal(sitesToJSON(sites))
	if err != nil {
		t.Fatal(err)
	}

	var got []flowworks.Site
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}

	if len(got) != 2 {
		t.Fatalf("want 2, got %d", len(got))
	}
	if got[0].SiteName != "Cavendish Creek" {
		t.Errorf("want Cavendish Creek, got %s", got[0].SiteName)
	}
}
