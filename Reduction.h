#include "Householder_reflector.h"

namespace Hessenberg_reduction {

template <typename Scalar>
class Reduction {
 public:
  Reduction(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
            bool find_conversion = false);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_hess_form() const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_conversion() const;

 private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hess_form;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> conversion;
};

template <typename Scalar>
Reduction<Scalar>::Reduction(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
    bool find_conversion) {
  int order = start.rows();
  if (find_conversion) {
    conversion =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(order,
                                                                        order);
  }
  for (int k = 0; k < order - 2; ++k) {
    Householder_reflector::Reflector reflector(start, k + 1, k);
    reflector.reflect_left(start, k + 1, order - 1, k, order - 1);
    reflector.reflect_right(start, 0, order - 1, k + 1, order - 1);
    if (find_conversion) {
      conversion(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) -=
          2  * (conversion(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) *
          reflector.get_u_vector()) * reflector.get_u_vector().transpose();
    }
  }
  hess_form = start;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Reduction<Scalar>::get_hess_form() const {
  return hess_form;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Reduction<Scalar>::get_conversion() const {
  return conversion;
}

};  // namespace Hessenberg_reduction

namespace Tridiagonal_reduction {

template <typename Scalar>
class Reduction {
 public:
  Reduction(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
            bool find_conversion = false);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& get_diag() const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& get_off_diag() const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_conversion()
      const;

 private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> off_diag;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> conversion;
};

template <typename Scalar>
Reduction<Scalar>::Reduction(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
    bool find_conversion) {
  int order = start.rows();
  if (find_conversion) {
    conversion =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(order,
                                                                        order);
  }
  for (int k = 0; k < order - 2; ++k) {
    Householder_reflector::Reflector reflector(start, k + 1, k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_ref, v_ref;
    u_ref = reflector.get_u_vector();
    v_ref = 2 * start(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) *
            u_ref;
    v_ref.tail(order - k - 1) -=
        u_ref * (u_ref.transpose() * v_ref.tail(order - k - 1));
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> uvt =
        u_ref * v_ref.transpose();
    start(Eigen::seq(k + 1, order - 1), Eigen::seq(0, order - 1)) -= uvt;
    start(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) -=
        uvt.transpose();
    if (find_conversion) {
      conversion(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) -=
          2  * (conversion(Eigen::seq(0, order - 1), Eigen::seq(k + 1, order - 1)) *
          reflector.get_u_vector()) * reflector.get_u_vector().transpose();
    }
  }
  diag = start.diagonal(0);
  off_diag.resize(order, 1);
  off_diag(0, 0) = {};
  off_diag.tail(order - 1) = start.diagonal(1);
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Reduction<Scalar>::get_diag()
    const {
  return diag;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&
Reduction<Scalar>::get_off_diag() const {
  return off_diag;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Reduction<Scalar>::get_conversion() const {
  return conversion;
}

};  // namespace Tridiagonal_reduction
