// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <string>
#include <vector>
#include <set>
#include <iostream>

#include "common.h"
#include "mesh.hpp"
#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

using namespace owl;

struct Texture {
    ~Texture()
    {
        if (pixel) delete[] pixel;
    }

    uint32_t* pixel{ nullptr };
    vec2i     resolution{ -1 };
};

struct Model {
    ~Model()
    {
        for (auto mesh : meshes) delete mesh;
    }

    std::vector<Mesh *> meshes;
    std::vector<Texture*>      textures;
    std::vector<vec3f>         vertices;
    //! bounding box of all vertices in the model
    box3f bounds;
};

Model* loadOBJ(const std::string& objFile, bool isLight);
#pragma once
