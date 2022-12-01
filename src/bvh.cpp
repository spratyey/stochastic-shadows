#include "bvh.hpp"

float evaluateSAHForLightBVH(LightBVH& node, std::vector<TriLight>& primitives, int axis, float pos)
{
    AABB leftBox, rightBox;
    int leftCount = 0, rightCount = 0;

    for (uint32_t i = node.primIdx; i < node.primCount; i++) {
        TriLight& light = primitives[i];

        if (light.cg[axis] < pos) {
            leftCount++;
            leftBox.grow(light.v1);
            leftBox.grow(light.v2);
            leftBox.grow(light.v3);
        }
        else {
            rightCount++;
            rightBox.grow(light.v1);
            rightBox.grow(light.v2);
            rightBox.grow(light.v3);
        }
    }

    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0.f ? cost : 1e30f;
}

int getLightBVHHeight(uint32_t nodeIdx, std::vector<LightBVH>& bvh)
{
    LightBVH& node = bvh[nodeIdx];
    if (node.primCount != 0) {
        return 0;
    }

    int leftHeight = getLightBVHHeight(node.left, bvh);
    int rightHeight = getLightBVHHeight(node.right, bvh);

    return max(leftHeight, rightHeight) + 1;
}