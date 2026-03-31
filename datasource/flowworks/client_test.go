package flowworks_test

import (
	"fmt"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skete-io/lark/datasource/flowworks"
)

// --- test server helpers ---

type mockServer struct {
	mux    *http.ServeMux
	server *httptest.Server
}

func newMockServer() *mockServer {
	mux := http.NewServeMux()
	s := httptest.NewServer(mux)
	return &mockServer{mux: mux, server: s}
}

func (m *mockServer) Close() { m.server.Close() }
func (m *mockServer) URL() string { return m.server.URL }

func (m *mockServer) handle(pattern string, fn http.HandlerFunc) {
	m.mux.HandleFunc(pattern, fn)
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"Token":    "test-token-abc",
		"IssuedAt": "2025-01-01 00:00:00.000",
		"Expires":  time.Now().UTC().Add(time.Hour).Format("2006-01-02 15:04:05.000"),
	})
}

// --- tests ---

func TestClient_Authenticate(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()

	ms.handle("/authenticate", authHandler)
	ms.handle("/sites", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-token-abc" {
			http.Error(w, "missing auth", http.StatusUnauthorized)
			return
		}
		writeJSON(w, map[string]interface{}{
			"Resources":     []interface{}{},
			"ResultCode":    0,
			"ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	_, err := c.Sites(context.Background())
	if err != nil {
		t.Fatalf("Sites error: %v", err)
	}
}

func TestClient_TokenRefreshOn401(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()

	var authCalls atomic.Int32
	ms.handle("/authenticate", func(w http.ResponseWriter, r *http.Request) {
		authCalls.Add(1)
		writeJSON(w, map[string]string{
			"Token":    "refreshed-token",
			"IssuedAt": "2025-01-01 00:00:00.000",
			"Expires":  time.Now().UTC().Add(time.Hour).Format("2006-01-02 15:04:05.000"),
		})
	})

	var siteCalls atomic.Int32
	ms.handle("/sites", func(w http.ResponseWriter, r *http.Request) {
		n := siteCalls.Add(1)
		// first call returns 401 to trigger refresh
		if n == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeJSON(w, map[string]interface{}{
			"Resources": []map[string]interface{}{
				{"SiteId": 1, "SiteName": "Test Site", "SiteType": "Sewer"},
			},
			"ResultCode": 0, "ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	sites, err := c.Sites(context.Background())
	if err != nil {
		t.Fatalf("Sites error: %v", err)
	}
	if len(sites) != 1 {
		t.Errorf("got %d sites, want 1", len(sites))
	}
}

func TestClient_ChannelData_ParsesFloat(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	ms.handle("/sites/241/channels/36843/data", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]interface{}{
			"Resources": []map[string]interface{}{
				{"DataValue": "1.23", "DataTime": "2024-01-01T08:00:00Z"},
				{"DataValue": "1.45", "DataTime": "2024-01-01T08:05:00Z"},
				{"DataValue": "1.67", "DataTime": "2024-01-01T08:10:00Z"},
			},
			"ResultCode": 0, "ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	pts, err := c.ChannelData(context.Background(), 241, 36843,
		flowworks.LastN("D", 1))
	if err != nil {
		t.Fatalf("ChannelData error: %v", err)
	}
	if len(pts) != 3 {
		t.Fatalf("got %d points, want 3", len(pts))
	}
	if pts[0].Value != 1.23 {
		t.Errorf("pts[0].Value = %.4f, want 1.23", pts[0].Value)
	}
	if pts[2].Value != 1.67 {
		t.Errorf("pts[2].Value = %.4f, want 1.67", pts[2].Value)
	}
}

func TestClient_ChannelData_NonNumericBecomesNaN(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	ms.handle("/sites/241/channels/36451/data", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]interface{}{
			"Resources": []map[string]interface{}{
				{"DataValue": "N/A", "DataTime": "2024-01-01T00:00:00Z"},
				{"DataValue": "0",   "DataTime": "2024-01-01T00:05:00Z"},
				{"DataValue": "1",   "DataTime": "2024-01-01T00:10:00Z"},
			},
			"ResultCode": 0, "ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	pts, err := c.ChannelData(context.Background(), 241, 36451,
		flowworks.LastN("D", 1))
	if err != nil {
		t.Fatalf("ChannelData error: %v", err)
	}
	if !math.IsNaN(pts[0].Value) {
		t.Errorf("pts[0].Value should be NaN for 'N/A', got %v", pts[0].Value)
	}
	if pts[0].Raw != "N/A" {
		t.Errorf("pts[0].Raw = %q, want 'N/A'", pts[0].Raw)
	}
}

func TestClient_ChannelData_Pagination(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	// count how many chunk requests are made
	var chunkCalls atomic.Int32
	ms.handle("/sites/241/channels/36843/data", func(w http.ResponseWriter, r *http.Request) {
		chunkCalls.Add(1)
		writeJSON(w, map[string]interface{}{
			"Resources": []map[string]interface{}{
				{"DataValue": "1.0", "DataTime": "2024-01-15T00:00:00Z"},
			},
			"ResultCode": 0, "ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	// 200-day range → should split into 3 chunks (90 + 90 + 20)
	pts, err := c.ChannelData(context.Background(), 241, 36843,
		flowworks.DateRange("2024-01-01", "2024-07-19"))
	if err != nil {
		t.Fatalf("ChannelData error: %v", err)
	}

	n := chunkCalls.Load()
	if n < 2 {
		t.Errorf("expected at least 2 chunk requests for 200-day range, got %d", n)
	}
	// each chunk returns 1 point, so len(pts) == number of chunks
	if len(pts) != int(n) {
		t.Errorf("pts len %d != chunk calls %d", len(pts), n)
	}
}

func TestClient_ChannelData_ContextCancellation(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	// slow handler — should be interrupted by context cancel
	ms.handle("/sites/241/channels/36843/data", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		writeJSON(w, map[string]interface{}{
			"Resources": []interface{}{}, "ResultCode": 0, "ResultMessage": "OK",
		})
	})

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	_, err := c.ChannelData(ctx, 241, 36843, flowworks.LastN("D", 1))
	if err == nil {
		t.Error("expected error on context timeout, got nil")
	}
}

func TestClient_MultiChannelData_AllChannels(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	for _, cid := range []string{"36843", "21881", "36451"} {
		id := cid // capture
		ms.handle("/sites/241/channels/"+id+"/data", func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, map[string]interface{}{
				"Resources": []map[string]interface{}{
					{"DataValue": "2.5", "DataTime": "2024-06-01T00:00:00Z"},
				},
				"ResultCode": 0, "ResultMessage": "OK",
			})
		})
	}

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	result, err := c.MultiChannelData(context.Background(), 241,
		[]int{36843, 21881, 36451}, flowworks.LastN("D", 30))
	if err != nil {
		t.Fatalf("MultiChannelData error: %v", err)
	}
	for _, cid := range []int{36843, 21881, 36451} {
		pts, ok := result[cid]
		if !ok {
			t.Errorf("missing channel %d in result", cid)
			continue
		}
		if len(pts) != 1 {
			t.Errorf("channel %d: got %d points, want 1", cid, len(pts))
		}
	}
}

func TestClient_APIError_NonZeroResultCode(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	ms.handle("/sites/241/channels/99999/data", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]interface{}{
			"Resources":     nil,
			"ResultCode":    5,
			"ResultMessage": "Channel not found",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	_, err := c.ChannelData(context.Background(), 241, 99999, flowworks.LastN("D", 1))
	if err == nil {
		t.Error("expected error for non-zero ResultCode")
	}
}

func TestClient_SiteChannels(t *testing.T) {
	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	ms.handle("/sites/1/channels", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]interface{}{
			"Resources": []map[string]interface{}{
				{"ChannelId": 1, "ChannelName": "Depth",    "Units": "m"},
				{"ChannelId": 2, "ChannelName": "Rainfall", "Units": "mm"},
				{"ChannelId": 3, "ChannelName": "Float",    "Units": ""},
			},
			"ResultCode": 0, "ResultMessage": "OK",
		})
	})

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	channels, err := c.SiteChannels(context.Background(), 1)
	if err != nil {
		t.Fatalf("SiteChannels error: %v", err)
	}
	if len(channels) != 3 {
		t.Errorf("got %d channels, want 3", len(channels))
	}
	if channels[0].ChannelName != "Depth" {
		t.Errorf("channels[0].Name = %q, want 'Depth'", channels[0].ChannelName)
	}
}

// TestClient_MultiSiteRouting verifies that requests are correctly routed
// to different site/channel combinations.
func TestClient_MultiSiteRouting(t *testing.T) {
	combos := []struct {
		siteID    int
		channelID int
		label     string
	}{
		{1, 10, "site1-depth"},
		{1, 11, "site1-rainfall"},
		{1, 12, "site1-switch"},
		{2, 20, "site2-depth"},
		{2, 21, "site2-rainfall"},
		{2, 22, "site2-switch"},
	}

	ms := newMockServer()
	defer ms.Close()
	ms.handle("/authenticate", authHandler)

	for _, s := range combos {
		path := fmt.Sprintf("/sites/%d/channels/%d/data", s.siteID, s.channelID)
		label := s.label
		ms.handle(path, func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, map[string]interface{}{
				"Resources": []map[string]interface{}{
					{"DataValue": "1.0", "DataTime": "2024-01-01T00:00:00Z"},
				},
				"ResultCode": 0, "ResultMessage": "OK",
			})
			_ = label
		})
	}

	c := flowworks.NewClient(ms.URL(), "user", "pass")
	for _, s := range combos {
		pts, err := c.ChannelData(context.Background(), s.siteID, s.channelID,
			flowworks.LastN("D", 1))
		if err != nil {
			t.Errorf("%s: error %v", s.label, err)
			continue
		}
		if len(pts) != 1 {
			t.Errorf("%s: got %d points, want 1", s.label, len(pts))
		}
	}
}
