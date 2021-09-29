/*
 *
 * nbody_render_gl.cpp
 *
 * Simple OpenGL renderer for n-body simulation.
 *
 * Copyright (c) 2019-2021, Uplink Laboratories, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "nbody_render_gl.h"

#if defined(USE_GL)

#include <chrono>
#include <thread>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>

#include <SDL.h>

#include "nbody_util.h"
static SDL_Window *g_window = NULL;
static SDL_GLContext g_context;

static GLuint g_vao = 0;
static GLuint g_vbo = 0;

static GLuint g_vertexShaderPoints = 0;
static GLuint g_fragmentShaderPoints = 0;
static GLuint g_programPoints = 0;

static GLint g_uModelViewMatrix = -2;
static GLint g_uProjectionMatrix = -2;

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
    "#version 330 core\n"
    "\n"
    "layout (location = 0) in vec3 aPos;\n"
    "\n"
    "uniform mat4 u_ModelViewMatrix;\n"
    "uniform mat4 u_ProjectionMatrix;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec4 vert = vec4(aPos, 1.0);\n"
    "    gl_Position = u_ProjectionMatrix * u_ModelViewMatrix * vert;\n"
    "}\n"
;

static const GLchar *s_fragmentShaderPoints =
    "#version 330 core\n"
    "\n"
    "out vec4 o_Color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    o_Color = vec4(1.0f);\n"
    "}\n"
;

static glm::mat4 g_ProjectionMatrix;
static glm::mat4 g_ModelViewMatrix;

static void _gl_resize(void)
{
    g_ProjectionMatrix = glm::perspective(glm::radians(70.0f), (float)g_width / (float)g_height, 0.1f, 1000.0f);
    g_ModelViewMatrix = glm::mat4(1.0f);
    glViewport(0, 0, g_width, g_height);
}

static bool check_shader_compile(GLint shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        int length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        if (length) {
            char *message = new char[length];
            glGetShaderInfoLog(shader, length, &length, message);
            printf("%s\n", message);
            delete [] message;
        }
    }
    return status == GL_TRUE;
}

static bool check_program_link(GLint program)
{
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        int length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        if (length) {
            char *message = new char[length];
            glGetProgramInfoLog(program, length, &length, message);
            printf("%s\n", message);
            delete [] message;
        }
    }
    return status == GL_TRUE;
}

int gl_init_window(void)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetSwapInterval(1);

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

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glGenBuffers(1, &g_vbo);

    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    g_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(g_vertexShaderPoints, 1, &s_vertexShaderPoints, 0);
    glCompileShader(g_vertexShaderPoints);
    assert(check_shader_compile(g_vertexShaderPoints));

    g_fragmentShaderPoints = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(g_fragmentShaderPoints, 1, &s_fragmentShaderPoints, 0);
    glCompileShader(g_fragmentShaderPoints);
    assert(check_shader_compile(g_fragmentShaderPoints));

    g_programPoints = glCreateProgram();
    glAttachShader(g_programPoints, g_vertexShaderPoints);
    glAttachShader(g_programPoints, g_fragmentShaderPoints);
    glLinkProgram(g_programPoints);

    assert(check_program_link(g_programPoints));

    g_uModelViewMatrix = glGetUniformLocation(g_programPoints, "u_ModelViewMatrix");
    g_uProjectionMatrix = glGetUniformLocation(g_programPoints, "u_ProjectionMatrix");

    _gl_resize();

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
    glUseProgram(g_programPoints);
    glUniformMatrix4fv(g_uModelViewMatrix, 1, GL_FALSE, glm::value_ptr(g_ModelViewMatrix));
    glUniformMatrix4fv(g_uProjectionMatrix, 1, GL_FALSE, glm::value_ptr(g_ProjectionMatrix));
    glBindVertexArray(g_vao);
    glDrawArrays(GL_POINTS, 0, g_N);
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

    for (c = 0; c < 3; c++) {
        g_camera_trans_lag[c] += (g_camera_trans[c] - g_camera_trans_lag[c]) * CAMERA_INERTIA;
        g_camera_rot_lag[c] += (g_camera_rot[c] - g_camera_rot_lag[c]) * CAMERA_INERTIA;
    }

    g_ModelViewMatrix = glm::translate(glm::vec3(g_camera_trans_lag[0], g_camera_trans_lag[1], g_camera_trans_lag[2]));
    g_ModelViewMatrix *= glm::rotate(glm::radians(g_camera_rot_lag[0]), glm::vec3(1.0f, 0.0f, 0.0f));
    g_ModelViewMatrix *= glm::rotate(glm::radians(g_camera_rot_lag[1]), glm::vec3(0.0f, 1.0f, 0.0f));
    g_ModelViewMatrix *= glm::rotate(glm::radians(g_camera_rot_lag[2]), glm::vec3(0.0f, 0.0f, 1.0f));

    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, g_N * 4 * sizeof(g_hostAOS_PosMass[0]), g_hostAOS_PosMass, GL_STREAM_DRAW);

    _gl_draw_points();

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
