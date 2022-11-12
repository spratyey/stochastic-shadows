#include "twoPassBsp.hpp"

TwoPassBSP::TwoPassBSP(std::vector<vec4f> &planes, vec3f minBound, vec3f maxBound) {
    // TODO: Implement custom lambda to consider epsilon
    std::set<vec4f> planeSet;
    for (auto &plane : planes) {
        if (planeSet.find(plane) == planeSet.end()) {
            this->planes.push_back(plane);
            planeSet.insert(plane);
        }
    }

    firstPass = new BSP(this->planes, minBound, maxBound);

    leaves.push_back(vec3f(0));

    for (int i = 1; i < firstPass->leaves.size(); i += 1) {
        points.push_back(std::make_pair(i, firstPass->leaves[i]));
    }

    std::pair<int, int> planeSpan = std::make_pair(0, this->planes.size());
    std::pair<int, int> pointSpan = std::make_pair(0, this->points.size());
    root = makeNode(planeSpan, pointSpan);

    delete firstPass;
}

int TwoPassBSP::makeNode(std::pair<int, int> &planeSpan, std::pair<int, int> &pointSpan) {
    int planeStart = planeSpan.first;
    int planeEnd = planeSpan.second;

    int pointStart = pointSpan.first;
    int pointEnd = pointSpan.second;

    int bestRating = 0;
    int bestIndex = -1;

    for (int i = planeStart; i < planeEnd; i++) {
        vec4f plane = planes[i];
        int rating = evaluateCut(plane, pointSpan);

        if (rating > bestRating) {
            bestRating = rating;
            bestIndex = planes.size();
        }

        if (rating != 0) {
            planes.push_back(plane);
        }
    }

    if (bestIndex < 0) {
        return makeLeaf(pointSpan);
    }

    vec4f bestPlane = planes[bestIndex];
    planes.erase(planes.begin() + bestIndex);

    std::pair<int, int> newPlaneSpan = std::make_pair(planeEnd, planes.size());
    return makeInnerNode(newPlaneSpan, pointSpan, bestPlane);
}

int TwoPassBSP::makeInnerNode(std::pair<int, int> &planeSpan, std::pair<int, int> &pointSpan, vec4f &plane) {
    depth += 1;

    int planeStart = planeSpan.first;
    int planeEnd = planeSpan.second;

    int pointStart = pointSpan.first;
    int pointEnd = pointSpan.second;

    BSPNode node = {};
    node.plane = plane;
    node.silSpan = vec2i(-1, -1);
    node.left = -1;
    node.right = -1;
    int nodeIndex = nodes.size();
    nodes.push_back(node);

    std::pair<int, int> newPointSpan = split(plane, pointSpan);
    node.left = makeNode(planeSpan, newPointSpan);
    vec4f negPlane(-plane.x, -plane.y, -plane.z, -plane.w);
    newPointSpan = split(negPlane, pointSpan);
    node.right = makeNode(planeSpan, newPointSpan);

    points.resize(pointStart);
    planes.resize(planeStart);

    nodes[nodeIndex] = node;

    depth -= 1;

    return nodeIndex;
}

int TwoPassBSP::makeLeaf(std::pair<int, int> &pointSpan) {
    int pointStart = pointSpan.first;
    int pointEnd = pointSpan.second;

    vec3f position(0);

    for (int i = pointStart; i < pointEnd; i++) {
        position += points[i].second;
    }

    position /= (pointEnd - pointStart);

    leaves.push_back(position);

    leafCount++;
    averageDepth += depth;

    return -(leaves.size() - 1);
}

bool TwoPassBSP::evaluateCut(vec4f &plane, std::pair<int, int> &pointSpan) {
    int pointStart = pointSpan.first;
    int pointEnd = pointSpan.second;

    int count1 = 0, count2 = 0;

    for (int i = pointStart; i < pointEnd; i++) {
        float distance = getPlanePointDist(points[i].second, plane);

        count1 += distance > 0 ? 1 : 0;
        count2 += distance < 0 ? 1 : 0;
    }

    return std::min(count1, count2);
}

std::pair<int, int> TwoPassBSP::split(vec4f &plane, std::pair<int,int> &pointSpan) {
    int pointStart = pointSpan.first;
    int pointEnd = pointSpan.second;
    int newPointStart = points.size();

    for (int i = pointStart; i < pointEnd; i++) {
        float distance = getPlanePointDist(points[i].second, plane);
        if (distance > 0) {
            points.push_back(points[i]);
        }
    }

    return std::make_pair(newPointStart, points.size());
}