#include <ik/utils.hpp>
#include <ik/chain_element.hpp>
#include <ik/initial_chain_element.hpp>
#include <ik/constant_chain_element.hpp>
#include <ik/variable_chain_element.hpp>
#include <ik/chain.hpp>

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

  using Chain = IK::ChainBuilder<T00>
  ::BuildVariable<T01>
  ::BuildConstant<T12>
  ::BuildVariable<T23>
  ::BuildConstant<T34>
  ::BuildVariable<T45>
  ::BuildConstant<T56>
  ::BuildVariable<T67>
  ::BuildConstant<T78>
  ;

  auto Q = Matrix<double, 4, 1>();
  auto P = Chain::forward(Q);

  cout << "begin" << endl;
  int offset = 0;
  for(int i = -10 ; i <= 10 ; i++) {
      Matrix<double, 4,1> T(12.0,(double)i,0.0,1.0);
      int count = 0;

      //while(norm(T-P) > 0.1) {
      while(norm(T-P) > 0.1 && count < 100) {
      //while(count < 20) {
          cout << "" << count+offset;
          cout << " " << norm(T-P);
          cout << " " << P(0,0);
          cout << " " << P(1,0);
          cout << " " << Q(0,0);
          cout << " " << Q(1,0);
          cout << " " << Q(2,0);
          cout << endl;

          auto dQ = Chain::inverseStep(T,Q);

          double coef = norm(T-P)*0.03;
          Q = Q+dQ*coef;

          P = Chain::forward(Q);
          count++;
        }
      offset += count;
    }
  cout << "end" << endl;

  return 0;
}
