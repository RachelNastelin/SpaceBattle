#ifndef GUI_H
#define GUI_H

#include <stdint.h>

// Screen size
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768

typedef struct color {
  uint8_t magenta;
  uint8_t cyan;
  uint8_t green;
  uint8_t red;
} color_t;

void gui_init();
void gui_update_display();
void gui_set_pixel(int x, int y, color_t color);
void gui_draw_circle(int center_x, int center_y, float radius, color_t color);
void gui_draw_ship(int center_c, int center_y);
void gui_draw_cannonballs(int center_c, int center_y);
void gui_fade(float scale);
void gui_shutdown();

#endif
