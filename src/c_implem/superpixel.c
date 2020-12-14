#include <stdio.h>
#include <stdlib.h>
#include "superpixel.h"
#include "list.h"

pSuperpixel new_superpixel(int label, pListPixels *pixel_idxs, pList *values){
    pSuperpixel element;
    int length = 0;

    element = malloc(sizeof(Superpixel));

    element->label = label;
    element->pixel_idxs = pixel_idxs;
    element->values = values;

    element->value_mean = compute_mean_list_return_len(values, &length);
    element->length = length;
    element->pos_mean = compute_mean_pos(pixel_idxs);


    return element;
}


pSuperpixel merge_superpixel(pSuperpixel sp1, pSuperpixel sp2, int label){
    sp1->label = label;

    merge_pixels_list(sp1->pixel_idxs, sp2->pixel_idxs);
    merge_list(sp1->values, sp2->values);

    sp1->value_mean = ((sp1->value_mean * sp1->length) + (
        sp2->value_mean * sp2->length) ) / (sp1->length + sp2->length);

    
    sp1->pos_mean->x = ((sp1->pos_mean->x * sp1->length) + (
        sp2->pos_mean->x * sp2->length) ) / (sp1->length + sp2->length);
    
    sp1->pos_mean->y = ((sp1->pos_mean->y * sp1->length) + (
        sp2->pos_mean->y * sp2->length) ) / (sp1->length + sp2->length);

    sp1->length += sp2->length;

    return sp1;
}
