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

#include "model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb/stb_image.h"

using namespace owl;

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

/*! load a texture (if not already loaded), and return its ID in the
    model's textures[] vector. Textures that could not get loaded
    return -1 */
int loadTexture(Model* model,
    std::map<std::string, int>& knownTextures,
    const std::string& inFileName,
    const std::string& modelPath)
{
    if (inFileName == "")
        return -1;

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto& c : fileName)
        if (c == '\\') c = '/';
    fileName = modelPath + "/" + fileName;

    vec2i res;
    int   comp;
    unsigned char* image = stbi_load(fileName.c_str(),
        &res.x, &res.y, &comp, STBI_rgb_alpha);
    int textureID = -1;
    if (image) {
        textureID = (int)model->textures.size();
        Texture* texture = new Texture;
        texture->resolution = res;
        texture->pixel = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
            mirrored along the y axis - mirror them here */
        for (int y = 0; y < res.y / 2; y++) {
            uint32_t* line_y = texture->pixel + y * res.x;
            uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
            int mirror_y = res.y - 1 - y;
            for (int x = 0; x < res.x; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        model->textures.push_back(texture);
    }
    else {
        LOG("Could not load texture from " << fileName);
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

Model* loadOBJ(const std::string& objFile, bool isLight)
{
    Model* model = new Model;

    const std::string modelDir
        = objFile.substr(0, objFile.rfind('/') + 1);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
        = tinyobj::LoadObj(&attributes,
            &shapes,
            &materials,
            &err,
            &err,
            objFile.c_str(),
            modelDir.c_str(),
            /* triangulate */true);
    if (!readOK) {
        throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
    const vec3f* normal_array = (const vec3f*)attributes.normals.data();
    const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();

    for (int i = 0; i < attributes.vertices.size(); i += 3) {
      model->vertices.push_back(
        vec3f(
          attributes.vertices[i],
          attributes.vertices[i+1],
          attributes.vertices[i+2]
        )
      );
    }

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    std::map<std::string, int>      knownTextures;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        // Get adjacent face IDs for all edges
        // Edge is defined by two vertex ids (lower, higher)
        std::map<std::pair<int, int>, std::vector<int>> edgeFaces;

        for (int materialID : materialIDs) {
            // TODO: Put the uniqueization step in the class instead of here
            Mesh* mesh = new Mesh;

            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                // Set face for all edges
                edgeFaces[std::make_pair(std::min(idx0.vertex_index, idx1.vertex_index), std::max(idx0.vertex_index, idx1.vertex_index))].push_back(faceID);
                edgeFaces[std::make_pair(std::min(idx1.vertex_index, idx2.vertex_index), std::max(idx1.vertex_index, idx2.vertex_index))].push_back(faceID);
                edgeFaces[std::make_pair(std::min(idx0.vertex_index, idx2.vertex_index), std::max(idx0.vertex_index, idx2.vertex_index))].push_back(faceID);

                vec3i vidx(mesh->vertex.size(), mesh->vertex.size() + 1, mesh->vertex.size() + 2);
                mesh->vertex.push_back(vertex_array[idx0.vertex_index]);
                mesh->vertex.push_back(vertex_array[idx1.vertex_index]);
                mesh->vertex.push_back(vertex_array[idx2.vertex_index]);
                mesh->index.push_back(vidx);

                vec3i nidx(mesh->normal.size(), mesh->normal.size() + 1, mesh->normal.size() + 2);
                mesh->normal.push_back(normal_array[idx0.normal_index]);
                mesh->normal.push_back(normal_array[idx1.normal_index]);
                mesh->normal.push_back(normal_array[idx2.normal_index]);

                mesh->faces.push_back(Face(
                  normal_array[idx0.normal_index],
                  normal_array[idx1.normal_index],
                  normal_array[idx2.normal_index],
                  vertex_array[idx0.vertex_index],
                  vertex_array[idx1.vertex_index],
                  vertex_array[idx2.vertex_index]
                ));

                // TODO: Check
                vec3i tidx(mesh->texcoord.size(), mesh->texcoord.size() + 1, mesh->texcoord.size() + 2);
                if (idx0.texcoord_index > 0)
                    mesh->texcoord.push_back(texcoord_array[idx0.texcoord_index]);
                else
                    mesh->texcoord.push_back(1);

                if (idx1.texcoord_index > 0)
                    mesh->texcoord.push_back(texcoord_array[idx1.texcoord_index]);
                else
                    mesh->texcoord.push_back(1);

                if (idx2.texcoord_index > 0)
                    mesh->texcoord.push_back(texcoord_array[idx2.texcoord_index]);
                else
                    mesh->texcoord.push_back(1);

                mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
                mesh->diffuseTextureID = loadTexture(model,
                    knownTextures,
                    materials[materialID].diffuse_texname,
                    modelDir);

                mesh->alpha = (const float)materials[materialID].shininess;
                mesh->alphaTextureID = loadTexture(model,
                    knownTextures,
                    materials[materialID].specular_highlight_texname,
                    modelDir);

                mesh->emit = (const vec3f&)materials[materialID].diffuse;
            }

            // Create all edges in the mesh
            if (isLight) {
                for (auto it : edgeFaces) {
                    Edge edge;
                    edge.adjVert1 = it.first.first;
                    edge.adjVert2 = it.first.second;
                    edge.vert1 = model->vertices[edge.adjVert1];
                    edge.vert2 = model->vertices[edge.adjVert2];
                    switch (it.second.size()) {
                        case 0:
                        // Isolated edge
                        edge.adjFace1 = -1;
                        edge.adjFace2 = -1;
                        edge.numAdjFace = 0;
                        break;
                        case 1:
                        // Boundary edge
                        edge.adjFace1 = it.second[0];
                        edge.adjFace2 = -1;
                        edge.numAdjFace = 1;
                        break;
                        default:
                        // Non boundary edge (non manifold meshes not supported)
                        edge.adjFace1 = it.second[0];
                        edge.adjFace2 = it.second[1];
                        edge.numAdjFace = 2;
                        break;
                    }
                    mesh->insertEdge(edge);
                }
            }

            if (mesh->vertex.empty()) {
                delete mesh;
            }
            else {
                for (auto idx : mesh->index) {
                    if (idx.x < 0 || idx.x >= (int)mesh->vertex.size() ||
                        idx.y < 0 || idx.y >= (int)mesh->vertex.size() ||
                        idx.z < 0 || idx.z >= (int)mesh->vertex.size()) {
                        LOG("invalid triangle indices");
                        throw std::runtime_error("invalid triangle indices");
                    }
                }

                model->meshes.push_back(mesh);
            }
        }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
        for (auto vtx : mesh->vertex)
            model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
}
