#ifndef BINOC_HYPERPOLY_LAYOUT_H
#define BINOC_HYPERPOLY_LAYOUT_H

#include <array>

// One queried bipolar edge has two spatial-ring neighbour bits (i,j) and one
// temporal-neighbour bit (t).  dual_contouring.cpp writes those neighbours to
// HP slots with this function, and all downstream provenance consumers use the
// same definition.  This makes the temporal role a producer/consumer contract
// instead of inferring it from an unrelated face_index / 2 convention.
namespace hyperpoly_layout {

enum class AxisRole : int {
    temporal_neighbour = 0,
    spatial_ring_j = 1,
    spatial_ring_i = 2,
};

constexpr int corner_index(int i, int j, int temporal_side) {
    return i + 2 * j + 4 * temporal_side;
}

constexpr int coordinate_bit(int corner, AxisRole axis) {
    switch (axis) {
        case AxisRole::temporal_neighbour: return (corner >> 2) & 1;
        case AxisRole::spatial_ring_j: return (corner >> 1) & 1;
        case AxisRole::spatial_ring_i: return corner & 1;
    }
    return -1;
}

// Cyclic order (00,10,11,01) on the selected parameter face.  The face pair
// for AxisRole::temporal_neighbour is exactly the tdir=0/1 pair populated by
// bipolar_edge_neighbor_search in dual_contouring.cpp.
constexpr std::array<int, 4> face_corners(AxisRole axis, int side) {
    if (axis == AxisRole::temporal_neighbour) {
        return side == 0
            ? std::array<int, 4>{corner_index(0, 0, 0), corner_index(1, 0, 0),
                                 corner_index(1, 1, 0), corner_index(0, 1, 0)}
            : std::array<int, 4>{corner_index(0, 0, 1), corner_index(1, 0, 1),
                                 corner_index(1, 1, 1), corner_index(0, 1, 1)};
    }
    if (axis == AxisRole::spatial_ring_j) {
        return side == 0
            ? std::array<int, 4>{0, 1, 5, 4}
            : std::array<int, 4>{2, 3, 7, 6};
    }
    return side == 0
        ? std::array<int, 4>{0, 2, 6, 4}
        : std::array<int, 4>{1, 3, 7, 5};
}

constexpr const char* axis_role_name(AxisRole axis) {
    switch (axis) {
        case AxisRole::temporal_neighbour: return "temporal_neighbour";
        case AxisRole::spatial_ring_j: return "spatial_ring_j";
        case AxisRole::spatial_ring_i: return "spatial_ring_i";
    }
    return "invalid";
}

static_assert(corner_index(0, 0, 0) == 0);
static_assert(corner_index(1, 1, 0) == 3);
static_assert(corner_index(0, 0, 1) == 4);
static_assert(corner_index(1, 1, 1) == 7);

}  // namespace hyperpoly_layout

#endif  // BINOC_HYPERPOLY_LAYOUT_H
