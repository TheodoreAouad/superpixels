#ifndef __SUPERPIXEL_H__
#define __SUPERPIXEL_H__

#include "list.h"

typedef struct Superpixel
{
    int label;
    pList values;
    pListPixels pixel_idxs;
    int length;

    float value_mean;
    pTwoTuplet pos_mean;

}Superpixel, *pSuperpixel;


pSuperpixel new_superpixel(int label, pListPixels *pixel_idxs, pList *values);
pSuperpixel merge_superpixel(pSuperpixel sp1, pSuperpixel sp2, int label);

#endif