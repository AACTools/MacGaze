#ifndef CMEDIAPIPE_H
#define CMEDIAPIPE_H

#include <stdint.h>
#include <stdbool.h>

// Forward-declare opaque handles
typedef void* MPFaceLandmarkerHandle;
typedef void* MPImageHandle;

// Result struct returned to Swift
typedef struct {
    int success;
    int landmark_count;
    float* landmarks;    // landmark_count * 3 floats (x, y, z)
    int has_transform;
    float transform[16]; // 4×4 column-major
    float detect_latency_ms;
} MPFaceLandmarkerResult;

// Lifecycle
MPFaceLandmarkerHandle cmp_face_landmarker_create(const char* model_path, int* out_status);
void cmp_face_landmarker_close(MPFaceLandmarkerHandle handle);

// Detection (video mode, takes raw RGB uint8 data)
int cmp_face_landmarker_detect_video(
    MPFaceLandmarkerHandle handle,
    const uint8_t* rgb_data,
    int width,
    int height,
    int64_t timestamp_ms,
    MPFaceLandmarkerResult* out_result
);

// Free result
void cmp_face_landmarker_free_result(MPFaceLandmarkerResult* result);

#endif /* CMEDIAPIPE_H */
