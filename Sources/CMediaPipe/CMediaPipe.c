#include "CMediaPipe.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

// ---- Struct definitions matching the INSTALLED dylib version (0.10.35) ----
// These match the Python ctypes definitions that ship with the same version.

// RunningMode — dylib uses GitHub source values (1-based)
#define RUNNING_MODE_IMAGE  1
#define RUNNING_MODE_VIDEO  2

// ImageFormat — dylib uses GitHub source values
#define IMAGE_FORMAT_SRGB   1
#define IMAGE_FORMAT_SRGBA  2

// BaseOptionsC — matches Python ctypes for mediapipe 0.10.x
struct BaseOptionsC {
    const char* model_asset_buffer;      // offset 0
    unsigned int model_asset_buffer_count; // offset 8
    const char* model_asset_path;        // offset 16 (8-byte aligned)
    int delegate;                         // offset 24
    int host_environment;                 // offset 28
    int host_system;                      // offset 32
    const char* host_version;             // offset 40 (8-byte aligned, padding at 36)
    const char* ca_bundle_path;           // offset 48
};
// sizeof = 56 bytes

// NormalizedLandmarkC
struct NormalizedLandmarkC {
    float x, y, z;
    bool has_visibility;
    float visibility;
    bool has_presence;
    float presence;
    const char* name;
};

// NormalizedLandmarksC
struct NormalizedLandmarksC {
    struct NormalizedLandmarkC* landmarks;
    uint32_t landmarks_count;
};

// MatrixC
struct MatrixC {
    uint32_t rows;
    uint32_t cols;
    float* data;
};

// FaceLandmarkerResultC
struct FaceLandmarkerResultC {
    struct NormalizedLandmarksC* face_landmarks;
    uint32_t face_landmarks_count;
    void* face_blendshapes;
    uint32_t face_blendshapes_count;
    struct MatrixC* facial_transformation_matrixes;
    uint32_t facial_transformation_matrixes_count;
};

// Callback type
typedef void (*result_callback_fn)(int, const struct FaceLandmarkerResultC*, void*, int64_t);

// FaceLandmarkerOptionsC
struct FaceLandmarkerOptionsC {
    struct BaseOptionsC base_options;
    int running_mode;
    int num_faces;
    float min_face_detection_confidence;
    float min_face_presence_confidence;
    float min_tracking_confidence;
    bool output_face_blendshapes;
    bool output_facial_transformation_matrixes;
    result_callback_fn result_callback;
};

// ImageProcessingOptionsC (minimal)
struct ImageProcessingOptionsC {
    int rotation_degrees;
    bool mirrored;
};

// ---- Opaque types ----
typedef void* MpFaceLandmarkerPtr;
typedef void* MpImagePtr;

// ---- Function signatures matching dylib v0.10.x (NO error_msg on most calls) ----
// Create: (options*, handle*) → status  [NO error_msg in dylib 0.10.x]
typedef int (*MpCreateFn)(struct FaceLandmarkerOptionsC*, void**);
// DetectForVideo: (handle, image, options*, timestamp, result*) → status
typedef int (*MpDetectVideoFn)(void*, void*, struct ImageProcessingOptionsC*, int64_t, struct FaceLandmarkerResultC*);
// DetectImage: (handle, image, options*, result*) → status
typedef int (*MpDetectImageFn)(void*, void*, struct ImageProcessingOptionsC*, struct FaceLandmarkerResultC*);
// CloseResult: (result*) → void
typedef void (*MpCloseResultFn)(struct FaceLandmarkerResultC*);
// Close: (handle) → status
typedef int (*MpCloseFn)(void*);
// ImageCreate: (format, w, h, data*, size, image*, error_msg*) → status
// NOTE: ImageCreate DOES have error_msg in dylib 0.10.x
typedef int (*MpImageCreateFn)(int, int, int, const uint8_t*, int, void**, char**);

// ---- Dynamic library state ----
static void* g_dylib = NULL;
static MpCreateFn g_create = NULL;
static MpDetectVideoFn g_detect_video = NULL;
static MpDetectImageFn g_detect_image = NULL;
static MpCloseResultFn g_close_result = NULL;
static MpCloseFn g_close = NULL;
static MpImageCreateFn g_image_create = NULL;

static int load_symbols(void) {
    if (g_dylib) return 0;

    const char* paths[] = {
        "Frameworks/libmediapipe.dylib",
        "./macgaze/Frameworks/libmediapipe.dylib",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_dylib = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_dylib) break;
    }
    if (!g_dylib) return -1;

    g_create = (MpCreateFn)dlsym(g_dylib, "MpFaceLandmarkerCreate");
    g_detect_video = (MpDetectVideoFn)dlsym(g_dylib, "MpFaceLandmarkerDetectForVideo");
    g_detect_image = (MpDetectImageFn)dlsym(g_dylib, "MpFaceLandmarkerDetectImage");
    g_close_result = (MpCloseResultFn)dlsym(g_dylib, "MpFaceLandmarkerCloseResult");
    g_close = (MpCloseFn)dlsym(g_dylib, "MpFaceLandmarkerClose");
    g_image_create = (MpImageCreateFn)dlsym(g_dylib, "MpImageCreateFromUint8Data");

    if (!g_create || !g_detect_video || !g_close_result || !g_close || !g_image_create) {
        return -2;
    }
    return 0;
}

// ---- Public API ----

MPFaceLandmarkerHandle cmp_face_landmarker_create(const char* model_path, int* out_status) {
    *out_status = load_symbols();
    if (*out_status != 0) return NULL;

    struct FaceLandmarkerOptionsC opts;
    memset(&opts, 0, sizeof(opts));
    opts.base_options.model_asset_path = model_path;
    opts.running_mode = RUNNING_MODE_VIDEO;   // 1 in dylib 0.10.x
    opts.num_faces = 1;
    opts.min_face_detection_confidence = 0.5f;
    opts.min_face_presence_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    opts.output_face_blendshapes = false;
    opts.output_facial_transformation_matrixes = true;

    void* handle = NULL;
    int status = g_create(&opts, &handle);
    *out_status = status;
    return (MPFaceLandmarkerHandle)handle;
}

void cmp_face_landmarker_close(MPFaceLandmarkerHandle handle) {
    if (handle && g_close) {
        g_close(handle);
    }
}

int cmp_face_landmarker_detect_video(
    MPFaceLandmarkerHandle handle,
    const uint8_t* rgb_data,
    int width,
    int height,
    int64_t timestamp_ms,
    MPFaceLandmarkerResult* out_result
) {
    memset(out_result, 0, sizeof(MPFaceLandmarkerResult));
    if (!handle || !g_detect_video) return -1;

    // Create MpImage (SRGB = 0 in dylib 0.10.x)
    void* image = NULL;
    char* error_msg = NULL;
    int img_status = g_image_create(
        IMAGE_FORMAT_SRGB,    // 0
        width, height,
        rgb_data,
        width * height * 3,
        &image, &error_msg
    );
    if (error_msg) { free(error_msg); }
    if (img_status != 0 || !image) return -2;

    // Run detection (VIDEO mode)
    struct ImageProcessingOptionsC ipo;
    memset(&ipo, 0, sizeof(ipo));

    struct FaceLandmarkerResultC result_c;
    memset(&result_c, 0, sizeof(result_c));

    int status = g_detect_video(handle, image, &ipo, timestamp_ms, &result_c);

    if (status != 0) {
        if (g_close_result) g_close_result(&result_c);
        return status;
    }

    // Extract landmarks
    if (result_c.face_landmarks_count > 0 && result_c.face_landmarks) {
        struct NormalizedLandmarksC* face = &result_c.face_landmarks[0];
        int count = face->landmarks_count;
        out_result->landmark_count = count;
        if (count > 0 && face->landmarks) {
            out_result->landmarks = (float*)malloc(count * 3 * sizeof(float));
            for (int i = 0; i < count; i++) {
                out_result->landmarks[i * 3 + 0] = face->landmarks[i].x;
                out_result->landmarks[i * 3 + 1] = face->landmarks[i].y;
                out_result->landmarks[i * 3 + 2] = face->landmarks[i].z;
            }
        }
    }

    // Extract transformation matrix (column-major 4×4)
    if (result_c.facial_transformation_matrixes_count > 0 && result_c.facial_transformation_matrixes) {
        struct MatrixC* mtx = &result_c.facial_transformation_matrixes[0];
        if (mtx->data && mtx->rows == 4 && mtx->cols == 4) {
            out_result->has_transform = 1;
            for (int i = 0; i < 16; i++) {
                out_result->transform[i] = mtx->data[i];
            }
        }
    }

    out_result->success = 1;
    if (g_close_result) g_close_result(&result_c);
    return 0;
}

void cmp_face_landmarker_free_result(MPFaceLandmarkerResult* result) {
    if (result->landmarks) {
        free(result->landmarks);
        result->landmarks = NULL;
    }
    result->landmark_count = 0;
}
