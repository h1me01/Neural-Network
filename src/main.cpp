#include "nn/network.h"

int main() {
    std::vector<XorData> data = {
        {{0, 0}, 0}, // false
        {{0, 1}, 1}, // true
        {{1, 0}, 1}, // true
        {{1, 1}, 0}  // false
    };

    auto i = new Input<2>();
    auto l1 = new Affine<4>(i, WeightsInit::He);
    auto a1 = new Activation<ReLU>(l1);
    auto l2 = new Affine<4>(a1, WeightsInit::He);
    auto a2 = new Activation<ReLU>(l2);
    auto l3 = new Affine<1>(a2, WeightsInit::He);
    auto a3 = new Activation<Sigmoid>(l3);

    Network network(                //
        100'000,                    // epochs
        2,                          // batch size
        new MSE(),                  // loss function
        new Adam(0.001),            // optimizer
        {i, l1, a1, l2, a2, l3, a3} // layers
    );

    network.train(data);

    std::cout << "\n================ Network Test ================\n\n";
    for(const auto &d : data) {
        std::cout << " Input=(" << d.get_data()[0]  //
                  << ", " << d.get_data()[1] << ")" //
                  << ", prediction=" << network.predict(d) << "\n";
    }

    return 0;
}
