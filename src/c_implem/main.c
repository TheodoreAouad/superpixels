#include <stdlib.h>
#include <time.h>
#include "list.h"
#include "superpixel.h"

#define SIZE 8

int main(int argc, char* argv) {
    int a1 = 1;
    int a2 = 2;
    float m1;
    pTwoTuplet px1;
    pTwoTuplet px2;
    clock_t start, end;
    double cpu_time_used;

    pList l1 = new_list();
    pList l2 = new_list();
    pList l3;

    l1 = add_element(l1, a2);
    l1 = add_element(l1, a2);
    l1 = add_element(l1, a1);

    l2 = add_element(l2, a1);
    l2 = add_element(l2, a1);
    l2 = add_element(l2, a2);

    print_list(l1);
    print_list(l2);

    // merge_list(l1, l2);
    // print_list(l1);

    // clear_list(l1);
    printf("%d\n", l1->value);
    // clear_list(l2);


    
    px1 = new_two_tuplet(1, 1);
    px2 = new_two_tuplet(3, 5);

    printf("%d\n", sizeof(TwoTuplet));
    printf("X=%f, Y=%f\n", px1->x, px1->y);
    printf("X=%f, Y=%f\n", px2->x, px2->y);

    pListPixels lp1 = new_pixels_list();
    pListPixels lp2 = new_pixels_list();

    lp1 = add_pixel_element(lp1, px1);
    lp1 = add_pixel_element(lp1, px2);
    lp1 = add_pixel_element(lp1, px2);
    print_pixels_list(lp1);

    lp2 = add_pixel_element(lp2, px2);
    lp2 = add_pixel_element(lp2, px2);
    lp2 = add_pixel_element(lp2, px1);
    print_pixels_list(lp2);

    // merge_pixels_list(lp1, lp2);
    // print_pixels_list(lp1);

    pSuperpixel sp1 = new_superpixel(1, lp1, l1);
    pSuperpixel sp2 = new_superpixel(2, lp2, l2);

    printf("\n");
    printf("Pixels:");
    print_pixels_list(sp1->pixel_idxs);
    printf("Values:");
    print_list(sp1->values);
    printf("%f\n", sp1->value_mean);
    printf("X=%f Y=%f\n", sp1->pos_mean->x, sp1->pos_mean->y);
    printf("len = %d\n", sp1->length);

    start = clock();
    merge_superpixel(sp1, sp2, 3);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\nTime spent: %f\n", cpu_time_used);


    printf("\n");
    printf("Pixels:");
    print_pixels_list(sp1->pixel_idxs);
    printf("Values:");
    print_list(sp1->values);
    printf("%f\n", sp1->value_mean);
    printf("X=%f Y=%f\n", sp1->pos_mean->x, sp1->pos_mean->y);
    printf("len = %d\n", sp1->length);

    clear_pixels_list(lp1);
    clear_list(l1);
    // clear_pixels_list(lp2);

    return 0;
}