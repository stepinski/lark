package swmm

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

type Conduit struct {
	Name      string
	FromNode  string
	ToNode    string
	LengthM   float64
	ManningsN float64
	ZUp       float64
	ZDown     float64
}

type XSection struct {
	LinkName  string
	Shape     string
	DiameterM float64
}

type Junction struct {
	Name      string
	ElevM     float64
	MaxDepthM float64
}

type ParseResult struct {
	Conduits  []Conduit
	XSections []XSection
	Junctions []Junction
}

func ParseFile(path string) (ParseResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return ParseResult{}, err
	}
	defer f.Close()

	var result ParseResult
	var section string

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			section = strings.ToUpper(strings.Trim(line, "[]"))
			continue
		}
		fields := strings.Fields(line)
		switch section {
		case "CONDUITS":
			if len(fields) < 8 {
				continue
			}
			result.Conduits = append(result.Conduits, Conduit{
				Name:      fields[0],
				FromNode:  fields[1],
				ToNode:    fields[2],
				LengthM:   parseF(fields[3]),
				ManningsN: parseF(fields[4]),
				ZUp:       parseF(fields[5]),
				ZDown:     parseF(fields[6]),
			})
		case "XSECTIONS":
			if len(fields) < 3 {
				continue
			}
			result.XSections = append(result.XSections, XSection{
				LinkName:  fields[0],
				Shape:     fields[1],
				DiameterM: parseF(fields[2]),
			})
		case "JUNCTIONS":
			if len(fields) < 3 {
				continue
			}
			result.Junctions = append(result.Junctions, Junction{
				Name:      fields[0],
				ElevM:     parseF(fields[1]),
				MaxDepthM: parseF(fields[2]),
			})
		}
	}
	return result, scanner.Err()
}

func parseF(s string) float64 {
	f, _ := strconv.ParseFloat(s, 64)
	return f
}
