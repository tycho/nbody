#include "nbody_render_gl.h"

#if defined(USE_GL)

#include <chrono>
#include <thread>

#include <GL/glew.h>

#include <SDL.h>
#include <GL/gl.h>

#include "nbody_util.h"

static SDL_Window *g_window = NULL;
static SDL_GLContext g_context;
static GLuint g_pbo = 0;
static GLuint g_vertexShaderPoints = 0;
static GLuint g_programPoints = 0;

static int g_key = 0;
static int g_glOK = 0;
static int g_width = 1024, g_height = 768;

// Desired camera state
static float g_camera_trans[]     = {0.0f, -2.0f, -150.0f};
static float g_camera_rot[]       = {0.0f, 0.0f, 0.0f};

// Current camera state (gradually moves toward non-lag versions)
static float g_camera_trans_lag[] = {0.0f, -2.0f, -150.0f};
static float g_camera_rot_lag[]   = {0.0f, 0.0f, 0.0f};

static const float CAMERA_INERTIA = 0.05f;

extern float *g_hostAOS_PosMass;
extern size_t g_N;

static const GLchar *s_vertexShaderPoints =
    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 vert = vec4(gl_Vertex.xyz, 1.0);                              \n"
    "    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;     \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "}                                                                      \n"
;

static void _gl_resize(void)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)g_width / (float)g_height, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, g_width, g_height);
}

int gl_init_window(void)
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetSwapInterval(1);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return 1;
    }

    g_window = SDL_CreateWindow(
        "n-body demo",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        g_width,
        g_height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    if (!g_window) {
        return 2;
    }

    g_context = SDL_GL_CreateContext(g_window);
    if (!g_context) {
        return 3;
    }

    glewInit();

    _gl_resize();

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glGenBuffers(1, &g_pbo);

    g_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(g_vertexShaderPoints, 1, &s_vertexShaderPoints, 0);
    glCompileShader(g_vertexShaderPoints);

    g_programPoints = glCreateProgram();
    glAttachShader(g_programPoints, g_vertexShaderPoints);
    glLinkProgram(g_programPoints);

    g_glOK = 1;

    return 0;
}

int gl_quit(void)
{
    SDL_Quit();
    return 0;
}

static void _gl_draw_points(void)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, g_pbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, g_N);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
}

static void _handle_sdl_events(void)
{
    SDL_Event e;

    if (!g_glOK)
        return;

    while (SDL_PollEvent(&e) != 0) {
        switch (e.type) {
        case SDL_QUIT:
            g_key = 'q';
            break;
        case SDL_KEYDOWN:
            g_key = e.key.keysym.sym;
            break;
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                g_width = e.window.data1;
                g_height = e.window.data2;
                _gl_resize();
            }
            break;
        case SDL_MOUSEMOTION:
            if (e.motion.state) {
                float dx = e.motion.xrel;
                float dy = e.motion.yrel;
                Uint32 button = e.motion.state;
                if (button == SDL_BUTTON_RMASK) {
                    // Translate
                    g_camera_trans[0] += dx / 100.0f;
                    g_camera_trans[1] -= dy / 100.0f;
                } else if (button == SDL_BUTTON_MMASK || button == (SDL_BUTTON_RMASK | SDL_BUTTON_LMASK)) {
                    // Zoom
                    g_camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(g_camera_trans[2]);
                } else if (button == SDL_BUTTON_LMASK) {
                    // Rotate
                    g_camera_rot[0] += dy / 5.0f;
                    g_camera_rot[1] += dx / 5.0f;
                }
            }
            break;
        }
    }
}

int gl_display(void)
{
    int c;

    if (!g_glOK)
        return 1;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _handle_sdl_events();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (c = 0; c < 3; c++) {
        g_camera_trans_lag[c] += (g_camera_trans[c] - g_camera_trans_lag[c]) * CAMERA_INERTIA;
        g_camera_rot_lag[c] += (g_camera_rot[c] - g_camera_rot_lag[c]) * CAMERA_INERTIA;
    }

    glTranslatef(g_camera_trans_lag[0], g_camera_trans_lag[1], g_camera_trans_lag[2]);
    glRotatef(g_camera_rot_lag[0], 1.0f, 0.0f, 0.0f);
    glRotatef(g_camera_rot_lag[1], 0.0f, 1.0f, 0.0f);

    glBindBuffer(GL_ARRAY_BUFFER, g_pbo);
    glBufferData(GL_ARRAY_BUFFER, g_N * 4 * sizeof(g_hostAOS_PosMass[0]), g_hostAOS_PosMass, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(1.0f);
    glUseProgram(g_programPoints);
    _gl_draw_points();
    glUseProgram(0);

    SDL_GL_SwapWindow(g_window);

    std::chrono::microseconds const_60fps(16666);
    std::this_thread::sleep_for(const_60fps);

    return 0;
}

int gl_getch(void)
{
    int key = g_key;
    g_key = 0;
    return key;
}

#else

int gl_init_window(void) { return 1; }
int gl_quit(void) { return 0; }
int gl_display(void) { return 1; }
int gl_getch(void) { return 0; }

#endif
/* vim: set ts=4 sts=4 sw=4 et: */
