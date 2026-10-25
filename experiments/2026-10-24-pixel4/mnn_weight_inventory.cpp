// Inspect serialized convolution storage without expanding quantized weights.
#include "MNN_generated.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 1;
    std::vector<char> data((std::istreambuf_iterator<char>(input)), {});
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data.data()), data.size());
    if (!MNN::VerifyNetBuffer(verifier)) return 1;
    const auto* net = MNN::GetNet(data.data());
    std::cout << "name\ttype\tinput_channels\toutput_channels\tgroups\tkernel\tfloat_elements\tquant_type\tquant_bits\tquant_bytes\n";
    for (const auto* op : *net->oplists()) {
        if (op->main_type() != MNN::OpParameter_Convolution2D) continue;
        const auto* conv = op->main_as_Convolution2D();
        const auto* c = conv->common();
        const auto* q = conv->quanParameter();
        std::cout << op->name()->str() << '\t' << MNN::EnumNameOpType(op->type())
                  << '\t' << c->inputCount() << '\t' << c->outputCount()
                  << '\t' << c->group() << '\t' << c->kernelX() * c->kernelY()
                  << '\t' << (conv->weight() ? conv->weight()->size() : 0)
                  << '\t' << (q ? q->type() : 0)
                  << '\t' << (q ? q->aMaxOrBits() : 0)
                  << '\t' << (q && q->buffer() ? q->buffer()->size() : 0) << '\n';
    }
}
