#include <stdio.h>
#include <stdlib.h>
#include "list.h"
#include "utils.h"


pList new_list(){
    return NULL;
}

pTwoTuplet new_two_tuplet(float x, float y){
    pTwoTuplet res;
    res = malloc(sizeof(TwoTuplet));

    if (res == NULL){
        printf("Could not create tuple (memory error).");
        exit(EXIT_FAILURE);
    }

    res->x = x;
    res->y = y;

    return res;

}


pList add_element(pList array, float value){
    pList new_element=malloc(sizeof(List));
    
    if (new_element == NULL){
        printf("Could not allocate memory.");
        exit(EXIT_FAILURE);
    }

    new_element->value = value;
    new_element->next = array;

    return new_element;
}

pList merge_list(pList array1, pList array2){
    while (array1->next != NULL){
        array1 = array1->next;
    }

    array1->next = malloc(sizeof(List));
    array1->next = array2;
    return array1;
}


pListPixels merge_pixels_list(pListPixels array1, pListPixels array2){
    while (array1->next != NULL){
        array1 = array1->next;
    }

    array1->next = malloc(sizeof(ListPixels));
    array1->next = array2;
    return array1;
}


void print_list(pList array){
    printf("[");
    while (array != NULL){
        printf("%f, ", array->value);
        array = array->next;
    }
    printf("]\n");
}


pList clear_list(pList array){
	if(array == NULL)
		return new_list();

	while(array != NULL){
        pList tmp = array;
		array = array->next;
        free(tmp);
    }

	return array;
}


float compute_mean_list_return_len(pList array, int *length){
    int i = 0;
    float res = 0;

    while (array != NULL){
        res += array->value;
        array = array->next;
        i++;
    }

    *length = i;
    
    return res / i;
}

float compute_mean_list(pList array){
    int *length = NULL;
    float res;
    length = malloc(sizeof(int));

    if (length == NULL)
        exit(EXIT_FAILURE);

    res = compute_mean_list_return_len(array, length);
    free(length);
    return res;
}

pTwoTuplet compute_mean_pos(pListPixels pixels_idxs){
    int i = 0;
    pTwoTuplet res;

    res = malloc(sizeof(TwoTuplet));

    while (pixels_idxs != NULL){
        res->x += pixels_idxs->value->x;
        res->y += pixels_idxs->value->y;
        pixels_idxs = pixels_idxs->next;
        i++;
    }

    res->x = res->x / i;
    res->y = res->y / i;

    return res;
}


pListPixels new_pixels_list(){
    return NULL;
}


pListPixels add_pixel_element(pListPixels array, pTwoTuplet value){
    pListPixels new_element=malloc(sizeof(ListPixels));
    
    if (new_element == NULL){
        printf("Could not allocate memory.");
        exit(EXIT_FAILURE);
    }

    new_element->value = value;
    new_element->next = array;

    return new_element;
}

void print_pixels_list(pListPixels array){
    printf("[");
    while (array != NULL){
        printf("(X:%f, Y: %f), ", array->value->x, array->value->y);
        array = array->next;
    }
    printf("]\n");
}


pListPixels clear_pixels_list(pListPixels array){
	if(array == NULL)
		return new_pixels_list();

	while(array != NULL){
        pListPixels tmp = array;
		array = array->next;
        free(tmp);
    }

	return array;
}
