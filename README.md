# Black Hole Simulation (Metal 4K Optimized)

A high-performance, real-time Schwarzschild black hole ray tracer written in pure Objective-C++ using Apple Metal. This project implements general relativistic ray marching to simulate the visual appearance of a black hole, including gravitational lensing, an accretion disk with Doppler beaming.    

![Black Hole Preview](blackhole.png)

## Features

*   **Real-time Ray Tracing**: Solves the geodesic equations for photons in curved spacetime to render accurate gravitational lensing.
*   **Physically Based Accretion Disk**: Volumetric rendering with **blackbody radiation** models and relativistic **Doppler color shifting** (redshift/blueshift).
*   **Gravitational Lensing**: Lenses the actual background sky (Milky Way), creating accurate Einstein rings.
*   **Zero Dependencies**: Written in pure Objective-C++ and Metal, requiring no external libraries (uses standard macOS frameworks).
*   **Interactive Camera**: Switch between an orbital view and a free-flight mode to explore the scene.
*   **Real-time HUD**: Live debug overlay showing camera coordinates, black hole position, and orbital parameters.
*   **Dynamic Controls**: Real-time slider to adjust accretion disk temperature (0K - 100,000K).
*   **Quality Presets**: Adjustable rendering settings (Low to Ultra) to balance performance and visual fidelity.

## Requirements

*   **macOS** (10.15+ recommended)
*   **Metal-compatible GPU** (Apple Silicon M1/M2/M3 highly recommended for 4K performance)
*   **Xcode Command Line Tools** (installed via `xcode-select --install`)

## Build Instructions

1.  Open your terminal and navigate to the project directory:
    ```bash
    cd /path/to/blackhole-sim
    ```

2.  Build the application using `make`:
    ```bash
    make
    ```
    This will compile the `main_4k.mm` file into an executable named `blackhole-sim-metal`.

## How to Run

After building, you can run the simulation explicitly or use the convenience Make target:

**Option 1: Run directly**
```bash
./blackhole-sim-metal
```

**Option 2: Build and Run**
```bash
make run-4k
```

## Controls

| Key / Action | Function |
| :--- | :--- |
| **Mouse Drag** | Rotate the camera (Orbit mode) or Look around (Free mode) |
| **Scroll / Up / Down**| Zoom in / out (Orbit mode) |
| **Z / X** | Zoom In / Out |
| **C** | Toggle Camera Mode (Orbit <-> Free Flight) |
| **F** | Toggle Fullscreen |
| **V** | Cycle Debug Views |
| **1 - 4** | Set Quality Preset (Low, Medium, High, Ultra) |
| **Esc** | Quit Application |

### Free Flight Controls (When in Free Mode)
| Key | Movement |
| :--- | :--- |
| **W / S** | Move Forward / Backward |
| **A / D** | Move Left / Right |
| **Q / E** | Move Down / Up |
| **Shift** | Boost movement speed |
| **K / L** | Roll Camera Left / Right |

### Simulation Controls
| Key | Action |
| :--- | :--- |
| **T / G** | Move Black Hole Up / Down (Y-Axis) |


## Technical Implementation Details

*   **Language**: Objective-C++ (`.mm`) mixing C++ logic with Objective-C Metal API calls.
*   **Renderer**: Custom Metal Compute Shader implementing ray marching through the Schwarzschild metric.
*   **Physics**: Photon trajectories are calculated using curved-spacetime geodesics.

## Customization

You can adjust the simulation parameters at the top of `main_4k.mm` in the `SimParams` struct:
*   `diskBoost`: Adjusts the brightness of the accretion disk.
*   `rin` / `rout`: Adjusts the inner and outer radius of the accretion disk.
*   `maxSteps`: Adjusts the ray-marching precision (also controlled by Quality presets).

### Background Image Customization

The simulation uses an environment map for the background stars (Einstein ring effect). You can customize this by:

1.  **Changing the Image**: 
    Place your desired image file (JPG/PNG) in the project directory.
    Open `main_4k.mm` and find the `initWithFrame` method (around line 597).
    Update the filename in the `loadSkyTexture` call:
    ```objective-c
    // Change "eso.jpg" to your filename
    [self loadSkyTexture:@"your_image.jpg"];
    ```

2.  **Adjusting the Field of View (Zoom)**:
    If your background image looks too zoomed in or out, you can tweak the mapping in the shader.
    Open `main_4k.mm` and locate the `sampleSky` function (around line 360).
    Change the `zoomFactor` value:
    ```objective-c
    // Lower values (e.g., 0.5) zoom in, higher values (e.g., 2.5) zoom out
    float zoomFactor = 2.5; 
    ```
    Play with this number until the background wraps nicely for your specific image.