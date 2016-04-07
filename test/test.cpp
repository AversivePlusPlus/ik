#include <ik/utils.hpp>
#include <ik/chain_element.hpp>
#include <ik/initial_chain_element.hpp>
#include <ik/constant_chain_element.hpp>
#include <ik/variable_chain_element.hpp>

#include <iostream>
using namespace std;

using G = CAS::General<double, double>;
template<int VAL> using C = G::Const::Integer<VAL>;

int main(int, char**) {

  using T00 = G::Space3D::Identity;
  using T01 = G::Space3D::RotationZ;
  using T12 = G::Space3D::Translation<C<5>>;
  using T23 = G::Space3D::RotationZ;
  using T34 = G::Space3D::Translation<C<5>>;
  using T45 = G::Space3D::RotationZ;
  using T56 = G::Space3D::Translation<C<5>>;
  using T67 = G::Space3D::RotationZ;
  using T78 = G::Space3D::Translation<C<5>>;

  using Model =  IK::ChainElement<
  IK::ConstantChainElement<T78,
  IK::VariableChainElement<T67,
  IK::ConstantChainElement<T56,
  IK::VariableChainElement<T45,
  IK::ConstantChainElement<T34,
  IK::VariableChainElement<T23,
  IK::ConstantChainElement<T12,
  IK::VariableChainElement<T01,
  IK::InitialChainElement<T00
  >>>>>>>>>>;

  auto Q = Matrix<double, 4, 1>();
  auto P = Model::forward(Q);

  cout << "begin" << endl;
  int offset = 0;
  for(int i = -10 ; i <= 10 ; i++) {
      Matrix<double, 4,1> T(12.0,(double)i,0.0,1.0);
      int count = 0;

      while(norm(T-P) > 0.1) {
      //while(norm(T-P) > 0.1 && count < 100) {
      //while(count < 20) {
          cout << "" << count+offset;
          cout << " " << norm(T-P);
          cout << " " << P(0,0);
          cout << " " << P(1,0);
          cout << " " << Q(0,0);
          cout << " " << Q(1,0);
          cout << " " << Q(2,0);
          cout << endl;

          auto dQ = Model::inverseStep(T,Q);

          double coef = norm(T-P)*0.0008;
          Q = Q+dQ*coef;

          P = Model::forward(Q);
          count++;
        }
      offset += count;
    }
  cout << "end" << endl;

  return 0;
}
