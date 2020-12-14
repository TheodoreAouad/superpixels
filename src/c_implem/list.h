#ifndef __LIST_H__
#define __LIST_H__


typedef struct TwoTuplet{
    float x;
    float y;
}TwoTuplet, *pTwoTuplet;


typedef struct ListPixels{
    pTwoTuplet value;
    struct ListPixels *next;
}ListPixels, *pListPixels;


typedef struct List{
    float value;
    struct List *next;
}List, *pList;


pList new_list();
pList add_element(pList array, float value);
pList merge_list(pList array1, pList array2);
void print_list(pList array);
pList clear_list(pList array);

// TwoTuplet new_two_tuplet(float x, float y);
pTwoTuplet new_two_tuplet(float x, float y);

pListPixels new_pixels_list();
pListPixels add_pixel_element(pListPixels array, pTwoTuplet value);
pListPixels merge_pixels_list(pListPixels array1, pListPixels array2);
void print_pixels_list(pListPixels array);
pListPixels clear_pixels_list(pListPixels array);

float compute_mean_list_return_len(pList array, int *length);
float compute_mean_list(pList array);
pTwoTuplet compute_mean_pos(pListPixels pixels_idxs);


#endif