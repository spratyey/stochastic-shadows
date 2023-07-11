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

void reorderLightBVH(std::vector<LightBVH> &input, std::vector<LightBVH> &output) {
    std::queue<LightBVH> q;
    std::queue<std::pair<int, int>> parent;
    q.push(input[0]);

    while (!q.empty()) {
        LightBVH curNode = q.front();
        q.pop();

        int parentIdx = output.size();
        output.push_back(curNode);
        if (!parent.empty()) {
            std::pair<int, int> par = parent.front();
            if (par.first == 0) {
                output[par.second].left = parentIdx;
            } else {
                output[par.second].right = parentIdx;
            }
            parent.pop();
        }
        if (curNode.left > 0) {
            q.push(input[curNode.left]);
            parent.push(std::make_pair(0, parentIdx));
        }

        if (curNode.right > 0) {
            q.push(input[curNode.right]);
            parent.push(std::make_pair(1, parentIdx));
        }
    }
}