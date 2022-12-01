#include <fstream>
#include <vector>

template<typename T>
void inline serialize_vector(std::vector<T> &vec, std::ofstream &stream) {
    size_t sz = vec.size();
    stream.write((char *)&sz, sizeof(size_t));

    for (int i = 0; i < vec.size(); i += 1) {
        stream.write((char *)&vec[i], sizeof(T));
    }
}

template<typename T>
void inline deserialize_vector(std::vector<T> &vec, std::ifstream &stream) {
    size_t sz;
    stream.read((char *)&sz, sizeof(size_t));

    for (int i = 0; i < sz; i += 1) {
        T item;
        stream.read((char *)&item, sizeof(T));
        vec.push_back(item);
    }
}