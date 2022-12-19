#include <iostream>
#include <string>

using namespace std;

#define MAX_LTC_LIGHTS 10

struct BST {
    int data;
    int left = -1;
    int right = -1;
};

int main()
{
    BST set[MAX_LTC_LIGHTS * 2];
    int setEnd = 0;

    int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
    int selectedEnd = 0;

    int areaLights[10] = { 1, 1, 2, 3, 4, 7, 3, 5, 8, 9 };
    int numAreaLights = 10;

    int ridx = 1;
    set[setEnd++].data = ridx;
    selectedIdx[selectedEnd++] = ridx;

    for (int i = 0; i < 10; i++) {
        ridx = areaLights[i];

        int setIdx = 0;
        bool found = false;
        for (int j = 0; j < MAX_LTC_LIGHTS; j++) {

            // If found
            if (set[setIdx].data == ridx) {
                found = true;
                break;
            }

            // Insert if empty node
            if (set[setIdx].data == -1) {
                set[setIdx].data = ridx;
                break;
            }

            // If child
            if (set[setIdx].left == -1 && set[setIdx].right == -1) {
                set[setEnd++].data = ridx;
                set[setEnd++].data = -1;

                if (ridx > set[setIdx].data) {
                    set[setIdx].right = setEnd - 2;
                    set[setIdx].left = setEnd - 1;
                }
                else {
                    set[setIdx].right = setEnd - 1;
                    set[setIdx].left = setEnd - 2;
                }

                break;
            }

            if (ridx > set[setIdx].data) {
                setIdx = set[setIdx].right;
            }
            else {
                setIdx = set[setIdx].left;
            }

        }

        if (!found)
            selectedIdx[selectedEnd++] = ridx;
    }

    for (int i = 0; i < selectedEnd; i++)
        std::cout << std::to_string(selectedIdx[i]) << std::endl;

	return 0;
}