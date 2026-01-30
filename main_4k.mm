// Black Hole Ray Tracer - Metal 4K Optimized
// Pure Objective-C++ implementation (no metal-cpp dependency)

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <chrono>
#include <cmath>
#include <iostream>
#import <simd/simd.h>

// ============================================================================
// Constants and Star Data (from ours.html)
// ============================================================================

constexpr float TAU = M_PI * 2.0f;

struct StarData {
  const char *name;
  float a_AU, e, i_deg, Omega_deg, omega_deg, Tp_year, P_year;
  simd_float3 color;
};

// Global State Declaration (needed for Camera access)
static simd_float3 g_blackHolePos = {0.0f, 0.0f, 0.0f};

const StarData STAR_S2 = {"S2",      0.1251f * 8178.0f, 0.8843f,
                          133.91f,   228.07f,           66.25f,
                          2018.379f, 16.0518f,          {0.70f, 0.85f, 1.00f}};
const StarData STAR_S55 = {"S55",
                           0.10424f * 8275.9f,
                           0.72980f,
                           159.59f,
                           319.43f,
                           327.77f,
                           2009.4738f,
                           12.22f,
                           {0.95f, 0.60f, 0.20f}};
const StarData STAR_S38 = {"S38",
                           0.14249f * 8275.9f,
                           0.81807f,
                           168.69f,
                           122.43f,
                           40.065f,
                           2022.6843f,
                           19.53f,
                           {0.75f, 0.30f, 0.95f}};
const StarData STAR_S29 = {"S29",
                           0.39025f * 8275.9f,
                           0.96880f,
                           144.24f,
                           4.9259f,
                           203.68f,
                           2021.4102f,
                           88.52f,
                           {0.20f, 0.95f, 0.30f}};
const StarData STAR_S4716 = {"S4716", 411.0f, 0.756f,
                             70.0f,   10.0f,  300.0f,
                             2022.5f, 4.02f,  {1.0f, 0.2f, 0.6f}};

// ============================================================================
// Uniform Structure
// ============================================================================

struct Uniforms {
  simd_float3 camPos;
  float time;
  simd_float3 camFwd;
  float rs;
  simd_float3 camRight;
  float rin;
  simd_float3 camUp;
  float rout;

  float tanHalfFov;
  float aspect;
  float dPhi;
  float escapeR;

  float diskBoost;
  int32_t maxSteps;
  float debugView;
  float qualityLevel;

  simd_float3 starPos;
  float starSize;
  simd_float3 starColor;
  float starBoost;

  simd_float3 star2Pos;
  float star2Size;
  simd_float3 star2Color;
  float star2Boost;

  simd_float3 star3Pos;
  float star3Size;
  simd_float3 star3Color;
  float star3Boost;

  simd_float3 star4Pos;
  float star4Size;
  simd_float3 star4Color;
  float star4Boost;

  simd_float3 star5Pos;
  float star5Size;
  simd_float3 star5Color;
  float star5Boost;

  float skyIntensity;
  float skyRotation;
  float diskBaseTemp;
  int32_t maxDiskCrossings;

  uint32_t width;
  uint32_t height;
  uint32_t _pad0, _pad1;
};

// ============================================================================
// Quality Presets
// ============================================================================

struct DisplayUniforms {
  float exposure;
  float contrast;
  float saturation;
  float bloomStrength;
  float vignetteStrength;
  float chromaticAberration;
  float grainStrength;
  float time;
};

enum class QualityPreset { Low, Medium, High, Ultra };

struct SimParams {
  float rs = 1.0f;
  float rin = 1.90f;
  float rout = 12.0f;
  int maxSteps = 200;
  float dPhi = 0.05f;
  float escapeR = 5000.0f;
  float diskBoost = 5.0f;
  float debugView = 0.0f;
  float starSize = 1.0f;
  float starBoost = 6.0f;

  float skyIntensity = 1.0f;
  float skyRotation = 0.0f;
  float diskBaseTemp = 8000.0f;
  int32_t maxDiskCrossings = 1;

  // Display Params (HDR)
  float exposure = 0.2f;
  float contrast = 1.0f;
  float saturation = 2.0f;
  float bloomStrength = 0.5f;
  float vignetteStrength = 0.5f;
  float chromaticAberration = 0.002f;
  float grainStrength = 0.0f; // Default Off

  QualityPreset quality = QualityPreset::High;

  void setQuality(QualityPreset q) {
    quality = q;
    switch (q) {
    case QualityPreset::Low:
      maxSteps = 100;
      dPhi = 0.08f;
      escapeR = 2000.0f;
      maxDiskCrossings = 1;
      break;
    case QualityPreset::Medium:
      maxSteps = 150;
      dPhi = 0.06f;
      escapeR = 3000.0f;
      maxDiskCrossings = 2;
      break;
    case QualityPreset::High:
      maxSteps = 200;
      dPhi = 0.05f;
      escapeR = 5000.0f;
      maxDiskCrossings = 3;
      break;
    case QualityPreset::Ultra:
      maxSteps = 300;
      dPhi = 0.04f;
      escapeR = 10000.0f;
      maxDiskCrossings = 8;
      break;
    }
  }
};

// ============================================================================
// Kepler Orbit Mathematics
// ============================================================================

float wrapTau(float x) {
  x = fmod(x, TAU);
  return x < 0 ? x + TAU : x;
}

float solveKepler(float M, float e) {
  float E = e < 0.8f ? M : M_PI;
  for (int k = 0; k < 8; k++) {
    float f = E - e * sin(E) - M;
    E -= f / (1.0f - e * cos(E));
  }
  return E;
}

simd_float3 orbitalPosition3D(const StarData &star, float year) {
  float n = TAU / star.P_year;
  float M = wrapTau(n * (year - star.Tp_year));
  float E = solveKepler(M, star.e);
  float r = star.a_AU * (1.0f - star.e * cos(E));
  float nu = 2.0f * atan2(sqrt(1.0f + star.e) * sin(E / 2.0f),
                          sqrt(1.0f - star.e) * cos(E / 2.0f));

  float xP = r * cos(nu), yP = r * sin(nu);
  float Omega = star.Omega_deg * M_PI / 180.0f;
  float i = star.i_deg * M_PI / 180.0f;
  float omega = star.omega_deg * M_PI / 180.0f;

  float x1 = xP * cos(omega) - yP * sin(omega);
  float y1 = xP * sin(omega) + yP * cos(omega);
  float x2 = x1, y2 = y1 * cos(i), z2 = y1 * sin(i);

  return simd_make_float3(x2 * cos(Omega) - y2 * sin(Omega),
                          x2 * sin(Omega) + y2 * cos(Omega), z2);
}

// ============================================================================
// Camera System
// ============================================================================

class Camera {
public:
  enum class Mode { Orbit, Free };
  Mode activeMode = Mode::Orbit;

  // phi = Pitch (up/down), theta = Yaw (around), roll = Bank (tilt)
  float distance = 15.0f, theta = 0.0f, phi = 1.48f, roll = 0.0f;
  simd_float3 freePos = {0.0f, 0.0f, 20.0f};
  float yaw = -M_PI / 2.0f, pitch = 0.0f, speed = 0.5f;

  bool keyW = false, keyA = false, keyS = false, keyD = false;
  bool keyQ = false, keyE = false, keyShift = false;
  bool keyZ = false, keyX = false;
  bool keyK = false, keyL = false;
  bool keyT = false, keyG = false;
  bool keyLeft = false, keyRight = false;
  bool isDragging = false;
  float lastMouseX = 0.0f, lastMouseY = 0.0f;

  simd_float3 getPosition() const {
    if (activeMode == Mode::Orbit) {
      return simd_make_float3(distance * sin(phi) * cos(theta),
                              distance * cos(phi),
                              distance * sin(phi) * sin(theta));
    }
    return freePos;
  }

  void update(float dt) {
    if (activeMode == Mode::Free) {
      float moveSpeed = speed * (keyShift ? 3.0f : 1.0f) * dt * 60.0f;
      simd_float3 fwd = {cos(yaw), 0.0f, sin(yaw)};
      simd_float3 right = {-sin(yaw), 0.0f, cos(yaw)};

      if (keyW)
        freePos += fwd * moveSpeed;
      if (keyS)
        freePos -= fwd * moveSpeed;
      if (keyD)
        freePos += right * moveSpeed;
      if (keyA)
        freePos -= right * moveSpeed;
      if (keyE)
        freePos.y += moveSpeed;
      if (keyQ)
        freePos.y -= moveSpeed;
    } else {
      // Orbit interaction
      float zoomSpeed = distance * 0.3f * dt;
      if (keyZ)
        distance = fmax(2.0f, distance - zoomSpeed);
      if (keyX)
        distance = fmin(5000.0f, distance + zoomSpeed);
    }

    // Roll Control (Available in both modes)
    float rollSpeed = 1.5f * dt;
    if (keyK)
      roll -= rollSpeed;
    if (keyL)
      roll += rollSpeed;

    // Black Hole Vertical Movement
    float bhMoveSpeed = 5.0f * dt;
    if (keyT)
      g_blackHolePos.y += bhMoveSpeed;
    if (keyG)
      g_blackHolePos.y -= bhMoveSpeed;

    // Orbit Control (Arrow Keys) matching mouse drag
    // Mouse drag uses theta += dx * 0.01f
    // Reduced speed for smoother control (was 1.5f)
    float orbitSpeed = 0.3f * dt;
    if (keyLeft)
      theta += orbitSpeed; // Similar to dragging right
    if (keyRight)
      theta -= orbitSpeed; // Similar to dragging left
  }

  void getVectors(simd_float3 &outPos, simd_float3 &outFwd,
                  simd_float3 &outRight, simd_float3 &outUp) const {
    outPos = getPosition();
    outFwd = (activeMode == Mode::Orbit)
                 ? simd_normalize(-outPos)
                 : simd_make_float3(cos(pitch) * cos(yaw), sin(pitch),
                                    cos(pitch) * sin(yaw));
    simd_float3 worldUp = {0.0f, 1.0f, 0.0f};

    // Base basis vectors
    simd_float3 baseRight = simd_normalize(simd_cross(outFwd, worldUp));
    simd_float3 baseUp = simd_cross(baseRight, outFwd);

    // Apply Roll Rotation
    // Rotate baseRight and baseUp around outFwd by 'roll' angle
    float c = cos(roll);
    float s = sin(roll);

    outRight = baseRight * c + baseUp * s;
    outUp = -baseRight * s + baseUp * c;
  }

  void onMouseDrag(float dx, float dy) {
    if (activeMode == Mode::Orbit) {
      theta += dx * 0.01f;
      phi = fmax(0.1f, fmin(M_PI - 0.1f, phi + dy * 0.01f));
    } else {
      yaw += dx * 0.003f;
      pitch = fmax(-1.47f, fmin(1.47f, pitch - dy * 0.003f));
    }
  }

  void onScroll(float delta) {
    if (activeMode == Mode::Orbit)
      distance = fmax(2.0f, fmin(5000.0f, distance * (1.0f - delta * 0.1f)));
  }

  void toggleMode() {
    if (activeMode == Mode::Orbit) {
      activeMode = Mode::Free;
      freePos = getPosition();
      yaw = atan2(-freePos.z, -freePos.x);
      pitch = asin(-freePos.y / simd_length(freePos));
    } else {
      activeMode = Mode::Orbit;
    }
  }
};

// ============================================================================
// Metal Shader Source
// ============================================================================

NSString *computeShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float3 camPos; float time;
    float3 camFwd; float rs;
    float3 camRight; float rin;
    float3 camUp; float rout;
    float tanHalfFov; float aspect; float dPhi; float escapeR;
    float diskBoost; int maxSteps; float debugView; float qualityLevel;
    float3 starPos; float starSize; float3 starColor; float starBoost;
    float3 star2Pos; float star2Size; float3 star2Color; float star2Boost;
    float3 star3Pos; float star3Size; float3 star3Color; float star3Boost;
    float3 star4Pos; float star4Size; float3 star4Color; float star4Boost;
    float3 star5Pos; float star5Size; float3 star5Color; float star5Boost;
    float skyIntensity; float skyRotation; float diskBaseTemp; int maxDiskCrossings;
    uint width; uint height; uint _pad0; uint _pad1;
};

inline float hash(float2 p) {
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

inline float noiseRaw(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(hash(i), hash(i + float2(1.0, 0.0)), u.x),
        mix(hash(i + float2(0.0, 1.0)), hash(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

inline float noise(float2 p, float lod) {
    float intensity = 1.0 - smoothstep(1.0, 4.0, lod);
    if (intensity < 0.01) return 0.5;
    return mix(0.5, noiseRaw(p), intensity);
}

inline float hash3(float3 p) {
    return fract(sin(dot(p, float3(12.9898, 78.233, 45.164))) * 43758.5453);
}

inline float noise3D(float3 p) {
    float3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + float3(1,0,0)), f.x),
                   mix(hash3(i + float3(0,1,0)), hash3(i + float3(1,1,0)), f.x), f.y),
               mix(mix(hash3(i + float3(0,0,1)), hash3(i + float3(1,0,1)), f.x),
                   mix(hash3(i + float3(0,1,1)), hash3(i + float3(1,1,1)), f.x), f.y), f.z);
}

inline float fbmDisk(float2 p, float lod) {
    float r = p.x / 12.0;
    float ang = p.y / 2.0;
    float ringScale = 2.0;
    float3 pos3D = float3(r * 12.0, cos(ang) * ringScale, sin(ang) * ringScale);
    
    float f = 0.0;
    f += 0.5 * noise3D(pos3D); pos3D *= 2.02;
    f += 0.25 * noise3D(pos3D); pos3D *= 2.03;
    f += 0.125 * noise3D(pos3D);
    
    return f;
}

inline float3 blackbodyColor(float t) {
    // Smooth gradient based on temperature chunks
    // To avoid banding, we use linear interpolation between control points.
    
    float3 col = float3(0.0);
    
    // Control points corresponding to realistic blackbody visual sequence
    const float3 c_1 = float3(0.8, 0.0, 0.0);   // 1000K - Dim Red
    const float3 c_2 = float3(1.0, 0.5, 0.0);   // 3000K - Orange
    const float3 c_3 = float3(1.0, 0.95, 0.8);  // 6000K - Warm White
    const float3 c_4 = float3(0.7, 0.8, 1.0);   // 10000K - Blue White
    const float3 c_5 = float3(0.4, 0.6, 1.0);   // 20000K+ - Deep Blue
    
    if (t < 1000.0) {
        // Fade out to black below 1000K
        col = mix(float3(0.0), c_1, t / 1000.0);
    } 
    else if (t < 3000.0) {
        col = mix(c_1, c_2, (t - 1000.0) / 2000.0);
    }
    else if (t < 6000.0) {
        col = mix(c_2, c_3, (t - 3000.0) / 3000.0);
    }
    else if (t < 10000.0) {
        col = mix(c_3, c_4, (t - 6000.0) / 4000.0);
    }
    else {
        // Cap transition at 25000K
        col = mix(c_4, c_5, saturate((t - 10000.0) / 15000.0));
    }
    
    return col;
}


inline float3 sampleSky(float3 rd, texture2d<float, access::sample> sky, 
                        float rotation, float intensity) {
    if (is_null_texture(sky)) return float3(0.0);

    // Rotate ray direction around Y axis
    float cosR = cos(rotation);
    float sinR = sin(rotation);
    float3 rotatedRd = float3(
        rd.x * cosR - rd.z * sinR,
        rd.y,
        rd.x * sinR + rd.z * cosR
    );
    
    // Spherical mapping with adjustable zoom
    float zoomFactor = 3.0;  // User requested param (0.2-0.5 recommended)
    float u = atan2(rotatedRd.z, rotatedRd.x) / (2.0 * 3.14159265) + 0.5;
    float v = asin(clamp(rotatedRd.y, -1.0, 1.0)) / 3.14159265 + 0.5;

    // Apply zoom (scales UV around center)
    u = (u - 0.5) * zoomFactor + 0.5;
    v = (v - 0.5) * zoomFactor + 0.5;
    
    constexpr sampler skySampler(filter::linear, address::repeat);
    // Use explicit LOD 0 to avoid derivative issues in divergent flow control
    float3 skyColor = sky.sample(skySampler, float2(u, v), level(0.0)).rgb;
    
    return skyColor * intensity;
}

kernel void blackHoleCompute(
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::sample> skyTexture [[texture(1)]],
    constant Uniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    
    float2 uv = float2(gid) / float2(u.width, u.height);
    float2 ndc = uv * 2.0 - 1.0;
    
    int dbg = int(u.debugView + 0.5);
    if (dbg == 1) { output.write(float4(uv, 0.0, 1.0), gid); return; }
    
    float3 rd = normalize(u.camFwd + u.camRight * (ndc.x * u.tanHalfFov * u.aspect) +
                          u.camUp * (ndc.y * u.tanHalfFov));
    float3 ro = u.camPos;
    
    if (dbg == 2) { output.write(float4(rd * 0.5 + 0.5, 1.0), gid); return; }
    
    float3 L = cross(ro, rd);
    float b = length(L);
    float3 Lhat = (b < 1e-5) ? normalize(cross(ro, float3(0.0, 1.0, 0.0))) : normalize(L);
    
    float3 e1 = normalize(ro - Lhat * dot(ro, Lhat));
    float3 e2 = cross(Lhat, e1);
    
    float r0 = length(ro);
    float u_orbit = 1.0 / max(r0, 1e-6);
    float3 dproj = normalize(rd - Lhat * dot(rd, Lhat));
    float w = -u_orbit * dot(dproj, e1) / max(dot(dproj, e2), 1e-4);
    float phi = 0.0;
    
    float3 pPrev = ro, p = ro;
    int hitType = 0;
    int diskCrossings = 0; // REPLACED: bool didHitDisk = false;
    float trans = 1.0;
    float3 accum = float3(0.0);
    
    for (int it = 0; it < u.maxSteps; it++) {
        float r = 1.0 / max(u_orbit, 1e-6);
        float h = u.dPhi * (r > 10.0 ? 2.0 : (r > 5.0 ? 1.5 : (r < 2.0 ? 0.7 : 1.0)));
        
        float k1w = 1.5 * u.rs * u_orbit * u_orbit - u_orbit;
        float u2 = u_orbit + 0.5 * h * w;
        float k2w = 1.5 * u.rs * u2 * u2 - u2;
        float u3 = u_orbit + 0.5 * h * (w + 0.5 * h * k1w);
        float k3w = 1.5 * u.rs * u3 * u3 - u3;
        float u4 = u_orbit + h * (w + 0.5 * h * k2w);
        float k4w = 1.5 * u.rs * u4 * u4 - u4;
        
        u_orbit += h * w + (h * h / 6.0) * (k1w + k2w + k3w);
        w += (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        phi += h;
        
        r = 1.0 / max(u_orbit, 1e-6);
        pPrev = p;
        p = (e1 * cos(phi) + e2 * sin(phi)) * r;
        
        if (r < u.rs * 1.001) { hitType = 1; break; }
        if (r > u.escapeR) { hitType = 3; break; }
        
        // MULTI-HIT LOGIC: Check counter instead of boolean
        if (pPrev.y * p.y < 0.0 && diskCrossings < u.maxDiskCrossings) {
            float t = pPrev.y / (pPrev.y - p.y);
            float3 phit = mix(pPrev, p, t);
            float rr = length(float2(phit.x, phit.z));
            
            if (rr > u.rin && rr < u.rout) {
                // 1. Calculate velocities and Doppler factor FIRST
                float3 vdir = normalize(float3(-phit.z, 0.0, phit.x));
                float vmag = min(sqrt(u.rs / (2.0 * rr)), 0.7);
                float3 rayDirAtHit = normalize(p - pPrev);
                float mu = dot(vdir, -rayDirAtHit);
                float gamma = 1.0 / sqrt(1.0 - vmag * vmag);
                float gravShift = sqrt(max(0.0, 1.0 - u.rs / rr)); // Gravitational Redshift
                float doppler = (1.0 / (gamma * (1.0 - vmag * mu))) * gravShift;
                
                // 2. Apply Doppler shift to temperature (System 1.5)
                float baseTemp = u.diskBaseTemp * pow(u.rin / rr, 0.75);
                float observedTemp = baseTemp * doppler;
                float3 blackbody = blackbodyColor(observedTemp);

                // 3. Render disk details
                float ang = atan2(phit.z, phit.x);
                float speed = 6.0 / sqrt(rr);
                float rotAngle = ang + u.time * speed * 0.2;
                float dist = length(phit - u.camPos);
                
                // Match JS LOD calculation
                float pixelSize = (u.tanHalfFov * dist * 2.0) / float(u.height);
                float texLOD = pixelSize * 40.0;
                
                float n = fbmDisk(float2(rr * 12.0, rotAngle * 2.0), texLOD);
                
                float filaments = smoothstep(0.2, 0.8, n);
                float detail = noise(float2(rr * 40.0, rotAngle * 4.0), texLOD * 3.0);
                filaments += detail * 0.2;

                float alphaInner = smoothstep(u.rin, u.rin + 1.0, rr);
                float alphaOuter = 1.0 - smoothstep(u.rout - 4.0, u.rout, rr);
                float alpha = alphaInner * alphaOuter;
                
                // 4. Combine
                float beamBase = max(doppler, 0.0);
                float beaming = beamBase * beamBase * beamBase; // D^3 remaining
                
                float3 diskCol = blackbody * filaments * beaming * u.diskBoost;
                
                // Reduce opacity for secondary hits to prevent overwashing
                float opacityMult = (diskCrossings == 0) ? 1.0 : 0.8; 
                float diskOpacity = saturate(alpha * 0.65 * opacityMult);
                
                accum += diskCol * diskOpacity * trans;
                trans *= (1.0 - diskOpacity);
                
                diskCrossings++; // Increment counter
                hitType = 2;
                
                if (trans < 0.01) break; // Early exit optimization
            }
        }
    }
    
    // Debug Mode 4: Visualize Escaped Ray Direction
    if (dbg == 3) {
        float4 col = (hitType == 2) ? float4(1,1,0,1) : (hitType == 1) ? float4(0,0,0,1) :
                     (hitType == 3) ? float4(0.1,0.3,1,1) : float4(0,0.6,0.2,1);
        output.write(col, gid);
        return;
    }

    if (hitType == 3 && trans > 0.01) {
        // Robustness Fix: Use numerical difference for direction
        // The analytical derivative (-w * r) becomes unstable at large r
        // because error in w is magnified by r. 
        // using (p - pPrev) is numerically stable and sufficient.
        
        float3 escapedRayDir = normalize(p - p); // Fallback
        if (length(p - pPrev) > 1e-6) {
            escapedRayDir = normalize(p - pPrev);
        } else {
             // Fallback for extremely small steps
             float3 radialDir = e1 * cos(phi) + e2 * sin(phi);
             float3 tangentDir = -e1 * sin(phi) + e2 * cos(phi);
             escapedRayDir = tangentDir;
        }
        
        // "Glass Stretch" Fix 2.0:
        // The user wants a MUCH sharper background far from the black hole.
        // The previous blend was too gentle (0.9), leaving residual "glassy" distortion.
        // We now enforce a hard transition back to the original ray direction 'rd'.
        
        float impactRatio = b / u.rs;
        
        // Transition range: [3.5 Rs, 6.0 Rs]
        // Below 3.5: Full gravitational lensing (Einstein ring preserved)
        // Above 6.0: Perfect straight-line transmission (Sharp background)
        float stabilityFactor = smoothstep(3.5, 6.0, impactRatio);
        
        // Fully mix to 'rd' (1.0) instead of 0.9 to eliminate ALL distortion in the far field.
        escapedRayDir = normalize(mix(escapedRayDir, rd, stabilityFactor)); 
        
        if (dbg == 4) {
             output.write(float4(escapedRayDir * 0.5 + 0.5, 1.0), gid);
             return;
        }
        
        float3 skyColor = sampleSky(escapedRayDir, skyTexture, u.skyRotation, u.skyIntensity);
        accum += skyColor * trans;
    }
    
    output.write(float4(accum, 1.0), gid);
}

kernel void gaussianBlur(
    texture2d<float, access::sample> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float2& direction [[buffer(0)]],
    constant float& threshold [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;
    
    // FIX: Use normalized coordinates for proper clamp_to_edge behavior on all GPUs
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    // CRITICAL FIX: Use OUTPUT dimensions for UV calculation to handle downsampling correctly!
    float2 size = float2(outputTexture.get_width(), outputTexture.get_height());
    float2 uv = (float2(gid) + 0.5) / size;
    float2 onePixel = 1.0 / size;
    
    // 9-tap Gaussian weights (sigma ~ 2.0)
    // 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
    
    float4 sum = float4(0.0);
    
    auto sample = [&](float2 coords) {
        float4 c = inputTexture.sample(s, coords);
        // Bright pass filter: Subtract threshold to remove dark areas from bloom
        c.rgb = max(float3(0.0), c.rgb - threshold); 
        return c;
    };
    
    sum += sample(uv) * 0.227027;
    
    float2 off1 = direction * 1.38461538 * onePixel;
    float2 off2 = direction * 3.23076923 * onePixel;
    
    sum += sample(uv + off1) * 0.3162162;
    sum += sample(uv - off1) * 0.3162162;
    sum += sample(uv + off2) * 0.0702703;
    sum += sample(uv - off2) * 0.0702703;
    
    outputTexture.write(sum, gid);
}

)";

NSString *displayShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut { float4 position [[position]]; float2 uv; };
constant float2 pos[] = { float2(-1,-1), float2(3,-1), float2(-1,3) };

struct DisplayUniforms {
    float exposure;
    float contrast;
    float saturation;
    float bloomStrength;
    float vignetteStrength;
    float chromaticAberration;
    float grainStrength;
    float time;
};

float hash12(float2 p) {
	float3 p3  = fract(float3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

inline float3 ACESFilm(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

inline float3 linearToSRGB(float3 linear) {
    float3 cutoff = step(linear, float3(0.0031308));
    float3 higher = 1.055 * pow(linear, 1.0/2.4) - 0.055;
    float3 lower = linear * 12.92;
    return mix(higher, lower, cutoff);
}

vertex VertexOut vertexMain(uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(pos[vid], 0.0, 1.0);
    out.uv = pos[vid] * 0.5 + 0.5;
    return out;
}

fragment float4 fragmentMain(VertexOut in [[stage_in]], 
                             texture2d<float> tex [[texture(0)]],
                             texture2d<float> bloomTex [[texture(1)]],
                             constant DisplayUniforms& u [[buffer(0)]]) {
    constexpr sampler s(filter::linear);
    
    // Chromatic Aberration
    float2 center = float2(0.5, 0.5);
    float2 dist = in.uv - center;
    float2 offset = dist * u.chromaticAberration;
    
    float3 hdr;
    hdr.r = tex.sample(s, in.uv - offset).r;
    hdr.g = tex.sample(s, in.uv).g;
    hdr.b = tex.sample(s, in.uv + offset).b;

    // float3 hdr = tex.sample(s, in.uv).rgb; // OLD
    float3 bloom = bloomTex.sample(s, in.uv).rgb;
    
    // Add Bloom (before tonemapping)
    hdr += bloom * u.bloomStrength;
    
    // 0. Vignette (Physical light reduction at edges)
    float2 uvCoords = in.uv * 2.0 - 1.0;
    float vignette = 1.0 - dot(uvCoords, uvCoords) * u.vignetteStrength;
    hdr *= saturate(vignette);
    
    // 1. Exposure
    hdr *= u.exposure;
    
    // 2. Contrast (around middle gray)
    float3 midGray = float3(0.18);
    hdr = midGray + (hdr - midGray) * u.contrast;
    hdr = max(hdr, 0.0);
    
    // 3. Saturation
    float luma = dot(hdr, float3(0.2126, 0.7152, 0.0722));
    hdr = luma + (hdr - luma) * u.saturation;
    hdr = max(hdr, 0.0);
    
    // 4. Tonemap (HDR -> LDR)
    float3 ldr = ACESFilm(hdr);
    
    // 5. Film Grain
    if (u.grainStrength > 0.0) {
        float noise = hash12(in.uv * u.time * 10.0);
        ldr += (noise * 2.0 - 1.0) * u.grainStrength;
    }
    
    // 6. Gamma Correction (Linear -> sRGB)
    float3 srgb = linearToSRGB(ldr);
    
    return float4(srgb, 1.0);
}
)";

// ============================================================================
// Global State
// ============================================================================

static Camera g_camera;
static SimParams g_params;
// static simd_float3 g_blackHolePos = {0.0f, 0.0f, 0.0f}; // Moved to top
static float g_simYear = 2024.0f;
static std::chrono::high_resolution_clock::time_point g_startTime;
static std::chrono::high_resolution_clock::time_point g_lastFrameTime;
static int g_frameCount = 0;
static float g_lastFPSTime = 0.0f;
static float g_currentFPS = 0.0f;

// ============================================================================
// Metal View
// ============================================================================

@interface BlackHoleView : MTKView
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLComputePipelineState> computePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> blurPipeline; // New
@property(nonatomic, strong) id<MTLRenderPipelineState> displayPipeline;
@property(nonatomic, strong) id<MTLBuffer> uniformBuffer;
@property(nonatomic, strong) id<MTLBuffer> displayUniformBuffer;
@property(nonatomic, strong) id<MTLTexture> renderTexture;
@property(nonatomic, strong) id<MTLTexture> bloomTextureA; // New: Ping
@property(nonatomic, strong) id<MTLTexture> bloomTextureB; // New: Pong
@property(nonatomic) uint32_t texWidth;
@property(nonatomic) uint32_t texHeight;
@property(nonatomic, strong) id<MTLTexture> skyTexture;
@property(nonatomic, strong) NSTextField *debugLabel;
@property(nonatomic, strong) NSSlider *tempSlider;
@property(nonatomic, strong) NSTextField *tempValueLabel;
@property(nonatomic, strong) NSSlider *exposureSlider;
@property(nonatomic, strong) NSTextField *exposureLabel;
@property(nonatomic, strong) NSSlider *satSlider;
@property(nonatomic, strong) NSTextField *satLabel;
@property(nonatomic, strong) NSSlider *bloomSlider;
@property(nonatomic, strong) NSTextField *bloomLabel;
@property(nonatomic, strong) NSSlider *grainSlider;
@property(nonatomic, strong) NSTextField *grainLabel;
@end

@implementation BlackHoleView

- (instancetype)initWithFrame:(NSRect)frame device:(id<MTLDevice>)device {
  self = [super initWithFrame:frame device:device];
  if (self) {
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.clearColor = MTLClearColorMake(0, 0, 0, 1);

    _commandQueue = [device newCommandQueue];

    NSError *error = nil;
    id<MTLLibrary> computeLib = [device newLibraryWithSource:computeShaderSource
                                                     options:nil
                                                       error:&error];
    if (!computeLib) {
      NSLog(@"Compute shader error: %@", error);
      return nil;
    }
    _computePipeline =
        [device newComputePipelineStateWithFunction:
                    [computeLib newFunctionWithName:@"blackHoleCompute"]
                                              error:&error];

    _blurPipeline = [device newComputePipelineStateWithFunction:
                                [computeLib newFunctionWithName:@"gaussianBlur"]
                                                          error:&error];

    id<MTLLibrary> displayLib = [device newLibraryWithSource:displayShaderSource
                                                     options:nil
                                                       error:&error];
    if (!displayLib) {
      NSLog(@"Display shader error: %@", error);
      return nil;
    }

    MTLRenderPipelineDescriptor *pipeDesc =
        [[MTLRenderPipelineDescriptor alloc] init];
    pipeDesc.vertexFunction = [displayLib newFunctionWithName:@"vertexMain"];
    pipeDesc.fragmentFunction =
        [displayLib newFunctionWithName:@"fragmentMain"];
    pipeDesc.colorAttachments[0].pixelFormat = self.colorPixelFormat;
    _displayPipeline = [device newRenderPipelineStateWithDescriptor:pipeDesc
                                                              error:&error];

    _uniformBuffer = [device newBufferWithLength:sizeof(Uniforms)
                                         options:MTLResourceStorageModeShared];

    _displayUniformBuffer =
        [device newBufferWithLength:sizeof(DisplayUniforms)
                            options:MTLResourceStorageModeShared];

    g_startTime = std::chrono::high_resolution_clock::now();
    g_lastFrameTime = g_startTime;

    // Load Sky Texture
    [self loadSkyTexture:@"eso.jpg"];

    // Initialize Debug Overlay
    _debugLabel =
        [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 300, 250)];
    [_debugLabel setEditable:NO];
    [_debugLabel setSelectable:NO];
    [_debugLabel setBezeled:NO];
    [_debugLabel setDrawsBackground:NO];
    [_debugLabel setTextColor:[NSColor whiteColor]];
    [_debugLabel setFont:[NSFont fontWithName:@"Menlo" size:12]];
    [_debugLabel setAlignment:NSTextAlignmentRight];
    [_debugLabel setStringValue:@"Initializing..."];
    [self addSubview:_debugLabel];

    // Initialize Temperature Slider (Standalone Debug UI)
    [self setupDebugUI];
  }
  return self;
}

// Standalone function for Debug UI (Easy to remove/disable)
- (void)setupDebugUI {
  // 1. Label
  _tempValueLabel =
      [[NSTextField alloc] initWithFrame:NSMakeRect(20, 60, 200, 20)];
  [_tempValueLabel setEditable:NO];
  [_tempValueLabel setSelectable:NO];
  [_tempValueLabel setBezeled:NO];
  [_tempValueLabel setDrawsBackground:NO];
  [_tempValueLabel setTextColor:[NSColor whiteColor]];
  [_tempValueLabel setFont:[NSFont systemFontOfSize:12]];
  [_tempValueLabel
      setStringValue:[NSString stringWithFormat:@"Temp: %.0f K",
                                                g_params.diskBaseTemp]];
  [self addSubview:_tempValueLabel];

  // 2. Slider
  _tempSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(20, 30, 200, 20)];
  [_tempSlider setMinValue:0.0];
  [_tempSlider setMaxValue:100000.0];
  [_tempSlider setFloatValue:g_params.diskBaseTemp];
  [_tempSlider setTarget:self];
  [_tempSlider setAction:@selector(onTempSliderChanged:)];
  [self addSubview:_tempSlider];

  // 3. Exposure Slider
  _exposureLabel =
      [[NSTextField alloc] initWithFrame:NSMakeRect(20, 100, 200, 20)];
  [_exposureLabel setEditable:NO];
  [_exposureLabel setSelectable:NO];
  [_exposureLabel setBezeled:NO];
  [_exposureLabel setDrawsBackground:NO];
  [_exposureLabel setTextColor:[NSColor whiteColor]];
  [_exposureLabel setStringValue:[NSString stringWithFormat:@"Exposure: %.1f",
                                                            g_params.exposure]];
  [self addSubview:_exposureLabel];

  _exposureSlider =
      [[NSSlider alloc] initWithFrame:NSMakeRect(20, 80, 200, 20)];
  [_exposureSlider setMinValue:0.1];
  [_exposureSlider setMaxValue:5.0];
  [_exposureSlider setFloatValue:g_params.exposure];
  [_exposureSlider setTarget:self];
  [_exposureSlider setAction:@selector(onExposureChanged:)];
  [self addSubview:_exposureSlider];

  // 4. Saturation Slider
  _satLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 150, 200, 20)];
  [_satLabel setEditable:NO];
  [_satLabel setSelectable:NO];
  [_satLabel setBezeled:NO];
  [_satLabel setDrawsBackground:NO];
  [_satLabel setTextColor:[NSColor whiteColor]];
  [_satLabel setStringValue:[NSString stringWithFormat:@"Saturation: %.1f",
                                                       g_params.saturation]];
  [self addSubview:_satLabel];

  _satSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(20, 130, 200, 20)];
  [_satSlider setMinValue:0.0];
  [_satSlider setMaxValue:2.0];
  [_satSlider setFloatValue:g_params.saturation];
  [_satSlider setTarget:self];
  [_satSlider setAction:@selector(onSatChanged:)];
  [self addSubview:_satSlider];

  // 5. Bloom Slider
  _bloomLabel =
      [[NSTextField alloc] initWithFrame:NSMakeRect(20, 200, 200, 20)];
  [_bloomLabel setEditable:NO];
  [_bloomLabel setSelectable:NO];
  [_bloomLabel setBezeled:NO];
  [_bloomLabel setDrawsBackground:NO];
  [_bloomLabel setTextColor:[NSColor whiteColor]];
  [_bloomLabel
      setStringValue:[NSString stringWithFormat:@"Bloom: %.1f",
                                                g_params.bloomStrength]];
  [self addSubview:_bloomLabel];

  _bloomSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(20, 180, 200, 20)];
  [_bloomSlider setMinValue:0.0];
  [_bloomSlider setMaxValue:2.0];
  [_bloomSlider setFloatValue:g_params.bloomStrength];
  [_bloomSlider setTarget:self];
  [_bloomSlider setAction:@selector(onBloomChanged:)];
  [self addSubview:_bloomSlider];

  // 6. Film Grain Slider
  _grainLabel =
      [[NSTextField alloc] initWithFrame:NSMakeRect(20, 250, 200, 20)];
  [_grainLabel setEditable:NO];
  [_grainLabel setSelectable:NO];
  [_grainLabel setBezeled:NO];
  [_grainLabel setDrawsBackground:NO];
  [_grainLabel setTextColor:[NSColor whiteColor]];
  [_grainLabel
      setStringValue:[NSString
                         stringWithFormat:@"Grain: %.2f",
                                          g_params.grainStrength / 0.04f]];
  [self addSubview:_grainLabel];

  _grainSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(20, 230, 200, 20)];
  [_grainSlider setMinValue:0.0];
  [_grainSlider setMaxValue:1.0];                              // UI scale 0..1
  [_grainSlider setFloatValue:g_params.grainStrength / 0.04f]; // Map back to UI
  [_grainSlider setTarget:self];
  [_grainSlider setAction:@selector(onGrainChanged:)];
  [self addSubview:_grainSlider];
}

- (void)onGrainChanged:(id)sender {
  float uiVal = [_grainSlider floatValue];
  g_params.grainStrength = uiVal * 0.04f; // Map 0..1 -> 0..0.04
  [_grainLabel
      setStringValue:[NSString stringWithFormat:@"Grain: %.2f", uiVal]];
}

- (void)onTempSliderChanged:(id)sender {
  float newVal = [_tempSlider floatValue];
  g_params.diskBaseTemp = newVal;
  [_tempValueLabel
      setStringValue:[NSString stringWithFormat:@"Temp: %.0f K", newVal]];
}

- (void)onExposureChanged:(id)sender {
  g_params.exposure = [_exposureSlider floatValue];
  [_exposureLabel setStringValue:[NSString stringWithFormat:@"Exposure: %.1f",
                                                            g_params.exposure]];
}

- (void)onSatChanged:(id)sender {
  g_params.saturation = [_satSlider floatValue];
  [_satLabel setStringValue:[NSString stringWithFormat:@"Saturation: %.1f",
                                                       g_params.saturation]];
}

- (void)onBloomChanged:(id)sender {
  g_params.bloomStrength = [_bloomSlider floatValue];
  [_bloomLabel
      setStringValue:[NSString stringWithFormat:@"Bloom: %.1f",
                                                g_params.bloomStrength]];
}

- (void)layout {
  [super layout];
  NSRect bounds = self.bounds;
  // Position in top right with 20px padding
  [_debugLabel setFrame:NSMakeRect(bounds.size.width - 320,
                                   bounds.size.height - 270, 300, 250)];
}

- (void)loadSkyTexture:(NSString *)filename {
  // Get path relative to executable or current directory
  NSString *path = [[NSFileManager defaultManager] currentDirectoryPath];
  path = [path stringByAppendingPathComponent:filename];

  NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
  if (!image) {
    NSLog(@"Failed to load sky texture: %@", path);
    return;
  }

  NSBitmapImageRep *bitmap = nil;
  for (NSImageRep *rep in [image representations]) {
    if ([rep isKindOfClass:[NSBitmapImageRep class]]) {
      bitmap = (NSBitmapImageRep *)rep;
      break;
    }
  }

  if (!bitmap) {
    // Fallback: create bitmap from image
    bitmap = [[NSBitmapImageRep alloc] initWithData:[image TIFFRepresentation]];
  }

  if (!bitmap) {
    NSLog(@"Failed to create bitmap from image: %@", path);
    return;
  }

  MTLTextureDescriptor *desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:bitmap.pixelsWide
                                  height:bitmap.pixelsHigh
                               mipmapped:YES];
  desc.usage = MTLTextureUsageShaderRead;

  _skyTexture = [self.device newTextureWithDescriptor:desc];

  if (!_skyTexture) {
    NSLog(@"Failed to create Metal texture from descriptor");
    return;
  }

  [_skyTexture
      replaceRegion:MTLRegionMake2D(0, 0, bitmap.pixelsWide, bitmap.pixelsHigh)
        mipmapLevel:0
          withBytes:bitmap.bitmapData
        bytesPerRow:bitmap.bytesPerRow];

  // Generate mipmaps
  id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
  [blit generateMipmapsForTexture:_skyTexture];
  [blit endEncoding];
  [cmdBuf commit];
  [cmdBuf waitUntilCompleted];

  std::cout << "Loaded sky texture: " << [filename UTF8String] << std::endl;
}

- (void)createTextureWithWidth:(uint32_t)width height:(uint32_t)height {
  if (_renderTexture && _texWidth == width && _texHeight == height)
    return;

  MTLTextureDescriptor *desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModePrivate;
  _renderTexture = [self.device newTextureWithDescriptor:desc];

  // Create Bloom Textures (Half Resolution)
  desc.width = width / 2;
  desc.height = height / 2;
  _bloomTextureA = [self.device newTextureWithDescriptor:desc];
  _bloomTextureB = [self.device newTextureWithDescriptor:desc];

  _texWidth = width;
  _texHeight = height;
}

// Helper to dispatch a blur pass
- (void)dispatchBlur:(id<MTLCommandBuffer>)cmdBuffer
                from:(id<MTLTexture>)src
                  to:(id<MTLTexture>)dst
                 dir:(simd_float2)direction
           threshold:(float)thresh {
  id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
  [enc setComputePipelineState:_blurPipeline];
  [enc setTexture:src atIndex:0];
  [enc setTexture:dst atIndex:1];
  [enc setBytes:&direction length:sizeof(direction) atIndex:0];
  [enc setBytes:&thresh length:sizeof(thresh) atIndex:1];

  MTLSize tgSize = MTLSizeMake(16, 16, 1);
  MTLSize tgCount =
      MTLSizeMake((dst.width + 15) / 16, (dst.height + 15) / 16, 1);

  [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
  [enc endEncoding];
}

- (void)drawRect:(NSRect)dirtyRect {
  auto now = std::chrono::high_resolution_clock::now();
  float dt = std::chrono::duration<float>(now - g_lastFrameTime).count();
  g_lastFrameTime = now;
  float time = std::chrono::duration<float>(now - g_startTime).count();

  g_camera.update(dt);

  CGSize size = self.drawableSize;
  [self createTextureWithWidth:(uint32_t)size.width
                        height:(uint32_t)size.height];

  simd_float3 camPos, camFwd, camRight, camUp;
  g_camera.getVectors(camPos, camFwd, camRight, camUp);

  const float ORBIT_SCALE = 40.0f;
  auto calcStarPos = [ORBIT_SCALE](const StarData &star) -> simd_float3 {
    simd_float3 p = orbitalPosition3D(star, g_simYear);
    return p / star.a_AU * ORBIT_SCALE;
  };

  Uniforms *u = (Uniforms *)_uniformBuffer.contents;
  // Adjust camera position relative to Black Hole for the shader
  // The shader assumes BH is at (0,0,0), so we shift the camera by -bhPos
  u->camPos = camPos - g_blackHolePos;
  u->time = time;
  u->camFwd = camFwd;
  u->rs = g_params.rs;
  u->camRight = camRight;
  u->rin = g_params.rin;
  u->camUp = camUp;
  u->rout = g_params.rout;
  u->tanHalfFov = tan((55.0f * M_PI / 180.0f) * 0.5f);
  u->aspect = size.width / size.height;
  u->dPhi = g_params.dPhi;
  u->escapeR = g_params.escapeR;
  u->diskBoost = g_params.diskBoost;
  u->maxSteps = g_params.maxSteps;
  u->debugView = g_params.debugView;
  u->qualityLevel = (float)g_params.quality;
  u->starPos = calcStarPos(STAR_S2);
  u->starSize = g_params.starSize;
  u->starColor = STAR_S2.color;
  u->starBoost = g_params.starBoost;
  u->star2Pos = calcStarPos(STAR_S55);
  u->star2Size = g_params.starSize * 0.9f;
  u->star2Color = STAR_S55.color;
  u->star2Boost = g_params.starBoost * 1.2f;
  u->star3Pos = calcStarPos(STAR_S38);
  u->star3Size = g_params.starSize * 1.1f;
  u->star3Color = STAR_S38.color;
  u->star3Boost = g_params.starBoost * 1.3f;
  u->star4Pos = calcStarPos(STAR_S29);
  u->star4Size = g_params.starSize * 0.95f;
  u->star4Color = STAR_S29.color;
  u->star4Boost = g_params.starBoost * 1.25f;
  u->star5Pos = calcStarPos(STAR_S4716);
  u->star5Size = g_params.starSize * 0.8f;
  u->star5Color = STAR_S4716.color;
  u->star5Boost = g_params.starBoost * 1.5f;
  u->skyIntensity = g_params.skyIntensity;
  u->skyRotation = g_params.skyRotation;
  u->diskBaseTemp = g_params.diskBaseTemp;
  u->maxDiskCrossings = g_params.maxDiskCrossings;
  u->width = _texWidth;
  u->height = _texHeight;

  id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];

  id<MTLComputeCommandEncoder> computeEncoder =
      [cmdBuffer computeCommandEncoder];
  [computeEncoder setComputePipelineState:_computePipeline];
  [computeEncoder setTexture:_renderTexture atIndex:0];
  if (_skyTexture) {
    [computeEncoder setTexture:_skyTexture atIndex:1];
  }
  [computeEncoder setBuffer:_uniformBuffer offset:0 atIndex:0];
  MTLSize tgSize = MTLSizeMake(16, 16, 1);
  MTLSize tgCount =
      MTLSizeMake((_texWidth + 15) / 16, (_texHeight + 15) / 16, 1);
  [computeEncoder dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
  [computeEncoder endEncoding];

  // --- BLOOM PASSES ---
  // Pass 1: Downsample + Blur Horz (RenderTexture -> BloomA)
  // WITH THRESHOLD: Cut off background starfield (brightness < 0.8) to fix
  // washed-out look.
  [self dispatchBlur:cmdBuffer
                from:_renderTexture
                  to:_bloomTextureA
                 dir:simd_make_float2(1, 0)
           threshold:0.8f];

  // Pass 2: Blur Vert (BloomA -> BloomB) - No threshold for subsequent passes
  [self dispatchBlur:cmdBuffer
                from:_bloomTextureA
                  to:_bloomTextureB
                 dir:simd_make_float2(0, 1)
           threshold:0.0f];

  // Pass 3: Blur Horz (BloomB -> BloomA)
  [self dispatchBlur:cmdBuffer
                from:_bloomTextureB
                  to:_bloomTextureA
                 dir:simd_make_float2(1, 0)
           threshold:0.0f];

  // Pass 4: Blur Vert (BloomA -> BloomB) - Final Result in BloomB
  [self dispatchBlur:cmdBuffer
                from:_bloomTextureA
                  to:_bloomTextureB
                 dir:simd_make_float2(0, 1)
           threshold:0.0f];
  // --------------------

  MTLRenderPassDescriptor *passDesc = self.currentRenderPassDescriptor;
  if (passDesc) {
    // Update Display Uniforms
    DisplayUniforms *du = (DisplayUniforms *)_displayUniformBuffer.contents;
    du->exposure = g_params.exposure;
    du->contrast = g_params.contrast;
    du->saturation = g_params.saturation;
    du->bloomStrength = g_params.bloomStrength;
    du->vignetteStrength = g_params.vignetteStrength;
    du->chromaticAberration = g_params.chromaticAberration;
    du->grainStrength = g_params.grainStrength;
    du->time = time;

    id<MTLRenderCommandEncoder> renderEncoder =
        [cmdBuffer renderCommandEncoderWithDescriptor:passDesc];
    [renderEncoder setRenderPipelineState:_displayPipeline];
    [renderEncoder setFragmentTexture:_renderTexture atIndex:0];
    [renderEncoder setFragmentTexture:_bloomTextureB
                              atIndex:1]; // Bind Final Bloom
    [renderEncoder setFragmentBuffer:_displayUniformBuffer offset:0 atIndex:0];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                      vertexStart:0
                      vertexCount:3];
    [renderEncoder endEncoding];
    [cmdBuffer presentDrawable:self.currentDrawable];
  }
  [cmdBuffer commit];

  g_frameCount++;
  float currentTime = std::chrono::duration<float>(now - g_startTime).count();
  if (currentTime - g_lastFPSTime > 1.0f) {
    g_currentFPS = g_frameCount / (currentTime - g_lastFPSTime);
    g_frameCount = 0;
    g_lastFPSTime = currentTime;
    const char *qn[] = {"Low", "Medium", "High", "Ultra"};
    std::cout << "FPS: " << (int)g_currentFPS << " | " << _texWidth << "x"
              << _texHeight << " | " << qn[(int)g_params.quality] << std::endl;
  }

  // Update Debug Overlay every frame
  NSString *debugInfo = [NSString
      stringWithFormat:@"--- CAMERA ---\n"
                       @"Pos: %.2f, %.2f, %.2f\n"
                       @"Dist: %.2f | Roll: %.2f\n"
                       @"Theta: %.2f | Phi: %.2f\n"
                       @"\n--- DISK ---\n"
                       @"Rs: %.2f | Boost: %.2f\n"
                       @"Rin: %.2f | Rout: %.2f\n"
                       @"\n--- SKY ---\n"
                       @"Int: %.2f | Rot: %.2f\n"
                       @"\n--- SYSTEM ---\n"
                       @"FPS: %d | Res: %dx%d\n"
                       @"\n--- BLACK HOLE ---\n"
                       @"Pos: %.2f, %.2f, %.2f",
                       camPos.x, camPos.y, camPos.z, g_camera.distance,
                       g_camera.roll, g_camera.theta, g_camera.phi, g_params.rs,
                       g_params.diskBoost, g_params.rin, g_params.rout,
                       g_params.skyIntensity, g_params.skyRotation,
                       (int)g_currentFPS, _texWidth, _texHeight,
                       g_blackHolePos.x, g_blackHolePos.y, g_blackHolePos.z];
  [_debugLabel setStringValue:debugInfo];
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)keyDown:(NSEvent *)event {
  unichar key = [[[event charactersIgnoringModifiers] lowercaseString]
      characterAtIndex:0];
  switch (key) {
  case 'w':
    g_camera.keyW = true;
    break;
  case 'a':
    g_camera.keyA = true;
    break;
  case 's':
    g_camera.keyS = true;
    break;
  case 'd':
    g_camera.keyD = true;
    break;
  case 'q':
    g_camera.keyQ = true;
    break;
  case 'e':
    g_camera.keyE = true;
    break;
  case 'z':
  case NSUpArrowFunctionKey:
    g_camera.keyZ = true;
    break;
  case 'x':
  case NSDownArrowFunctionKey:
    g_camera.keyX = true;
    break;
  case NSLeftArrowFunctionKey:
    g_camera.keyLeft = true;
    break;
  case NSRightArrowFunctionKey:
    g_camera.keyRight = true;
    break;
  case 'k':
    g_camera.keyK = true;
    break;
  case 'l':
    g_camera.keyL = true;
    break;
  case 't':
    g_camera.keyT = true;
    break;
  case 'g':
    g_camera.keyG = true;
    break;
  case 'c':
    g_camera.toggleMode();
    break;
  case 'v':
    g_params.debugView = fmod(g_params.debugView + 1.0f, 4.0f);
    break;
  case 'f':
    [self.window toggleFullScreen:nil];
    break;
  case '1':
    g_params.setQuality(QualityPreset::Low);
    std::cout << "Quality: Low" << std::endl;
    break;
  case '2':
    g_params.setQuality(QualityPreset::Medium);
    std::cout << "Quality: Medium" << std::endl;
    break;
  case '3':
    g_params.setQuality(QualityPreset::High);
    std::cout << "Quality: High" << std::endl;
    break;
  case '4':
    g_params.setQuality(QualityPreset::Ultra);
    std::cout << "Quality: Ultra" << std::endl;
    break;
  case 27:
    [NSApp terminate:nil];
    break;
  }
}

- (void)keyUp:(NSEvent *)event {
  unichar key = [[[event charactersIgnoringModifiers] lowercaseString]
      characterAtIndex:0];
  switch (key) {
  case 'w':
    g_camera.keyW = false;
    break;
  case 'a':
    g_camera.keyA = false;
    break;
  case 's':
    g_camera.keyS = false;
    break;
  case 'd':
    g_camera.keyD = false;
    break;
  case 'q':
    g_camera.keyQ = false;
    break;
  case 'e':
    g_camera.keyE = false;
    break;
  case 'z':
  case NSUpArrowFunctionKey:
    g_camera.keyZ = false;
    break;
  case 'x':
  case NSDownArrowFunctionKey:
    g_camera.keyX = false;
    break;
  case NSLeftArrowFunctionKey:
    g_camera.keyLeft = false;
    break;
  case NSRightArrowFunctionKey:
    g_camera.keyRight = false;
    break;
  case 'k':
    g_camera.keyK = false;
    break;
  case 'l':
    g_camera.keyL = false;
    break;
  case 't':
    g_camera.keyT = false;
    break;
  case 'g':
    g_camera.keyG = false;
    break;
  }
}

- (void)flagsChanged:(NSEvent *)event {
  g_camera.keyShift = ([event modifierFlags] & NSEventModifierFlagShift) != 0;
}

- (void)mouseDown:(NSEvent *)event {
  g_camera.isDragging = true;
  NSPoint loc = [event locationInWindow];
  g_camera.lastMouseX = loc.x;
  g_camera.lastMouseY = loc.y;
}

- (void)mouseUp:(NSEvent *)event {
  g_camera.isDragging = false;
}

- (void)mouseDragged:(NSEvent *)event {
  if (!g_camera.isDragging)
    return;
  NSPoint loc = [event locationInWindow];
  g_camera.onMouseDrag(loc.x - g_camera.lastMouseX,
                       loc.y - g_camera.lastMouseY);
  g_camera.lastMouseX = loc.x;
  g_camera.lastMouseY = loc.y;
}

- (void)scrollWheel:(NSEvent *)event {
  g_camera.onScroll([event deltaY]);
}

@end

// ============================================================================
// App Delegate
// ============================================================================

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) BlackHoleView *metalView;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Metal not supported!");
    [NSApp terminate:nil];
    return;
  }

  NSLog(@"Metal device: %@", device.name);

  NSScreen *screen = [NSScreen mainScreen];
  NSRect screenFrame = [screen visibleFrame];
  CGFloat width = fmin(3840, screenFrame.size.width * 0.9);
  CGFloat height = fmin(2160, screenFrame.size.height * 0.9);
  CGFloat x = (screenFrame.size.width - width) / 2 + screenFrame.origin.x;
  CGFloat y = (screenFrame.size.height - height) / 2 + screenFrame.origin.y;

  _window = [[NSWindow alloc]
      initWithContentRect:NSMakeRect(x, y, width, height)
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                          NSWindowStyleMaskResizable |
                          NSWindowStyleMaskMiniaturizable
                  backing:NSBackingStoreBuffered
                    defer:NO];
  [_window setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
  [_window setTitle:@"Black Hole - Metal 4K"];

  _metalView =
      [[BlackHoleView alloc] initWithFrame:NSMakeRect(0, 0, width, height)
                                    device:device];
  [_window setContentView:_metalView];
  [_window makeFirstResponder:_metalView];
  [_window makeKeyAndOrderFront:nil];

  std::cout << "\n============ Black Hole - Metal 4K ============\n";
  std::cout << "Quality: HIGH (60+ FPS @ 4K)\n\n";
  std::cout << "Controls:\n";
  std::cout << "  Drag: Rotate | Scroll: Zoom | F: Fullscreen\n";
  std::cout << "  C: Free camera | WASD/QE: Move | V: Debug\n";
  std::cout << "  1-4: Quality presets | Esc: Quit\n";
  std::cout << "================================================\n\n";
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  return YES;
}

@end

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;

  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];

    AppDelegate *delegate = [[AppDelegate alloc] init];
    [app setDelegate:delegate];

    NSMenu *menuBar = [[NSMenu alloc] init];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    [menuBar addItem:appMenuItem];
    NSMenu *appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"Quit"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];
    [app setMainMenu:menuBar];

    [app activateIgnoringOtherApps:YES];
    [app run];
  }
  return 0;
}
