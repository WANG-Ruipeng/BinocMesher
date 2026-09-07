#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <iostream>
#include "build/binocmesher/source/runtime_identity.h"
using namespace runtime_identity;
using VID = std::array<std::pair<int,int>,2>;
static_assert(4LL*kMaximumVertices < INT_MAX);
static_assert(11LL*kMaximumOwners < INT_MAX);
static_assert(3LL*kMaximumFaces < INT_MAX);
void shifts(int e,int n0,int g0,int n1,int g1) {
    record_sort_shift(e,0,g0); record_sort_shift(e,1,n0);
    record_sort_shift(e,2,g1); record_sort_shift(e,3,n1);
}
void finish() {
    std::vector<std::array<int,3>> empty;
    for(int e=0;e<state.elements;++e)
        finalize_element(e,empty,static_cast<int>(state.vertices[e].size()));
    complete();
}
int main() {
    enable(true); begin(true,true,false,5);
    shifts(0,100,7,200,9); shifts(4,300,11,500,13);
    const VID normalized{{{4,2},{8,3}}}; const VID unchanged=normalized;
    record_vertex(0,normalized,0); record_vertex(4,normalized,0);
    assert(normalized==unchanged);
    finish(); assert(state.status==ready);
    int rows[4]{}; assert(output_vertices(0,rows,4)==0);
    assert(rows[0]==104 && rows[1]==9 && rows[2]==208 && rows[3]==12);
    assert(output_vertices(4,rows,4)==0);
    assert(rows[0]==304 && rows[1]==13 && rows[2]==508 && rows[3]==16);
    int offsets[20]{}; assert(output_source_vid_shifts(offsets,19)==-1);
    assert(output_source_vid_shifts(offsets,20)==0);
    assert(offsets[0]==100 && offsets[1]==7 && offsets[2]==200 && offsets[3]==9);
    assert(offsets[16]==300 && offsets[17]==11 && offsets[18]==500 && offsets[19]==13);
    for(int i=4;i<16;++i) assert(offsets[i]==0);
    enable(true); begin(true,true,false,1);
    record_sort_shift(0,0,7); record_vertex(0,normalized,0);
    assert(state.status==error); assert(output_source_vid_shifts(offsets,20)==-1);
    enable(true); begin(true,true,false,1);
    record_sort_shift(0,0,7); record_sort_shift(0,0,7); assert(state.status==error);
    enable(true); begin(true,true,false,1); shifts(0,INT_MAX,0,0,0);
    record_vertex(0,VID{{{1,0},{0,0}}},0); assert(state.status==error);
    enable(true); begin(true,true,false,1); shifts(0,INT_MIN,0,0,0);
    record_vertex(0,VID{{{-1,0},{0,0}}},0); assert(state.status==error);
    enable(true); begin(true,true,false,1);
    record_sort_shift(0,4,0); assert(state.status==error);
    enable(true); begin(true,true,false,5); finish();
    assert(state.status==ready); assert(output_source_vid_shifts(offsets,20)==0);
    for(int i:offsets) assert(i==0);
    for(int e=0;e<5;++e) assert(count(e,false)==0);
    enable(true); begin(true,true,false,1); shifts(0,1,2,3,4);
    enable(false); assert(state.status==disabled);
    for(auto& element:state.source_vid_shifts) for(int x:element) assert(x==0);
    enable(true); begin(false,true,false,1); shifts(0,1,2,3,4);
    assert(state.status==unsupported);
    std::cout << "PASS: nonzero shifts, column mapping, multi-element, input immutability, "
                 "missing/duplicate/invalid pass, signed overflow, capacity, reset, empty, unsupported\n";
}
