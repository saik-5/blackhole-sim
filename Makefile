# Black Hole Metal Renderer - Makefile
# For macOS with Apple Silicon or Intel

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-deprecated-declarations
OBJCXXFLAGS = -fobjc-arc

# Frameworks
FRAMEWORKS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore -framework Foundation -framework Cocoa

.PHONY: all clean run-4k help

all: blackhole-sim-metal

help:
	@echo "Black Hole Metal Renderer"
	@echo ""
	@echo "  make           Build the 4K optimized version"
	@echo "  make run-4k    Build and run"
	@echo "  make clean     Remove built files"
	@echo ""
	@echo "Controls (when running):"
	@echo "  Drag mouse: Rotate camera"
	@echo "  Scroll: Zoom in/out"
	@echo "  F: Toggle fullscreen"
	@echo "  C: Switch camera mode (orbit/free)"
	@echo "  WASD/QE: Move (in free camera mode)"
	@echo "  1-4: Quality presets"
	@echo "  Esc: Quit"

# 4K optimized version (no external dependencies!)
blackhole-sim-metal: main_4k.mm
	@echo "Building Black Hole 4K..."
	$(CXX) $(CXXFLAGS) $(OBJCXXFLAGS) -o $@ $< $(FRAMEWORKS)
	@echo ""
	@echo "Build complete! Run with: ./blackhole-sim-metal"
	@echo "Or use: make run-4k"

clean:
	rm -f blackhole-sim-metal

run-4k: blackhole-sim-metal
	./blackhole-sim-metal
