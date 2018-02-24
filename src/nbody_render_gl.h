#ifndef __included_nbody_render_gl_h
#define __included_nbody_render_gl_h

#ifdef __cplusplus
extern "C" {
#endif

int gl_init_window(void);
int gl_display(void);
int gl_getch(void);
int gl_quit(void);

#ifdef __cplusplus
}
#endif

#endif
