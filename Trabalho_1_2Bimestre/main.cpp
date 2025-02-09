#include <iostream>
#include <vector>
#include <memory>

class Neuronio {   //class abst
protected:
    std::vector<double> entradas;
    std::vector<double> pesos;
    double bias;
public:
    Neuronio(const std::vector<double>& entradas, const std::vector<double>& pesos, double bias)
        : entradas(entradas), pesos(pesos), bias(bias) {}
    virtual double predict() = 0; 
    virtual ~Neuronio() {}
};

class NeuronioReLU : public Neuronio {   //classe deriv
public:
    NeuronioReLU(const std::vector<double>& entradas, const std::vector<double>& pesos, double bias)
        : Neuronio(entradas, pesos, bias) {}
    
    double predict() override {
        double soma = bias;
        for (size_t i = 0; i < entradas.size(); i++) {
            soma += entradas[i] * pesos[i];
        }
        return std::max(0.0, soma); 
    }
};

int main() {
    //mais testes
    std::vector<std::vector<double>> entradas = {
        {0.3, 2.0},  // Caso 1
        {0.3, 2.0},  // Caso 2
        {1.0, -0.5}, // Caso 3 (teste extra)
        {0.0, 0.0},  // Caso 4 (teste extra - saída sempre 0)
        {2.0, -3.0}  // Caso 5 (teste extra - pesos negativos)
    };

    std::vector<std::vector<double>> pesos = {
        {0.2, 0.4}, 
        {0.2, 0.4}, 
        {0.5, -0.2}, 
        {0.0, 0.0}, 
        {0.1, -0.5}
    };

    std::vector<double> bias = {-1.5, -0.5, 0.0, -2.0, 1.0};

    std::vector<std::unique_ptr<Neuronio>> neuronios;
    for (size_t i = 0; i < entradas.size(); i++) {
        neuronios.push_back(std::make_unique<NeuronioReLU>(entradas[i], pesos[i], bias[i]));
    }
    
    for (size_t i = 0; i < neuronios.size(); i++) {
        std::cout << "Caso " << i+1 << " - Saída: " << neuronios[i]->predict() << std::endl;
    }
    
    return 0;
}
