#include <cmath>

#include "Givens_rotation.h"
#include "Reduction.h"

namespace Single_shift {

template <typename Scalar>
class Single_shift {
 public:
  Single_shift(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
               Scalar eps, bool find_eigenvectors = false);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& get_eigenvalues() const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_eigenvectors() const;

 private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eigenvalues;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;
  Scalar wilkinson_shift(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& diag,
                         Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& off_diag,
                         int ind) const;
};

template <typename Scalar>
Single_shift<Scalar>::Single_shift(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start, Scalar eps,
    bool find_eigenvectors) {
  int order = start.rows();
  Tridiagonal_reduction::Reduction reduction(start, find_eigenvectors);
  if (find_eigenvectors) {
    eigenvectors =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(order,
                                                                        order);
  }
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag = reduction.get_diag();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> off_diag = reduction.get_off_diag();
  int ind = order - 1;
  while (ind > 0) {
    Scalar shift = wilkinson_shift(diag, off_diag, ind);
    Scalar x = diag(0) - shift;
    Scalar y = off_diag(1);
    for (int k = 0; k < ind; ++k) {
      Givens_rotation::Givens_rotation<Scalar> givens;
      if (ind > 1) {
        givens = Givens_rotation::Givens_rotation(x, y);
      } else {
        givens = Givens_rotation::Givens_rotation(diag(0), off_diag(1));
      }
      givens.rotate_tridiag(diag(k), diag(k + 1), off_diag(k + 1));
      if (k > 0) {
        off_diag(k) = givens.get_cos() * x - givens.get_sin() * y;
      }
      if (k < ind - 1) {
        y = -givens.get_sin() * off_diag(k + 2);
        off_diag(k + 2) = givens.get_cos() * off_diag(k + 2);
      }
      x = off_diag(k + 1);
      if (find_eigenvectors) {
        eigenvectors(Eigen::seq(0, order - 1), Eigen::seq(k, k + 1)) *=
            givens.right_rotation();
      }
    }
    Scalar sm = std::abs(diag(ind)) + std::abs(diag(ind - 1));
    if (std::abs(off_diag(ind)) < eps * sm) {
      --ind;
    }
  }
  eigenvectors = reduction.get_conversion() * eigenvectors;
  eigenvalues = diag;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&
Single_shift<Scalar>::get_eigenvalues() const {
  return eigenvalues;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Single_shift<Scalar>::get_eigenvectors() const {
  return eigenvectors;
}

template <typename Scalar>
Scalar Single_shift<Scalar>::wilkinson_shift(
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& diag,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& off_diag, int ind) const {
  Scalar d = (diag(ind - 1) - diag(ind)) / 2;
  if (diag(ind) == diag(ind - 1)) {
    return diag(ind) - std::abs(off_diag(ind));
  } else if (diag(ind) > diag(ind - 1)) {
    return diag(ind) -
           off_diag(ind) * off_diag(ind) / (d - std::hypot(d, off_diag(ind)));
  } else {
    return diag(ind) -
           off_diag(ind) * off_diag(ind) / (d + std::hypot(d, off_diag(ind)));
  }
}

};  // namespace Single_shift

namespace Double_shift {

template <typename Scalar>
class Double_shift {
 public:
  Double_shift(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
               Scalar eps, bool find_schur_vectors = false);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_triang()
      const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
  get_schur_vectors() const;

 private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> triang;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> schur_vectors;
};

template <typename Scalar>
Double_shift<Scalar>::Double_shift(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start, Scalar eps,
    bool find_schur_vectors) {
  int order = start.rows();
  Hessenberg_reduction::Reduction reduction(start, find_schur_vectors);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hess_form =
      reduction.get_hess_form();
  if (find_schur_vectors) {
    schur_vectors =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(order,
                                                                        order);
  }
  int p = order - 1;
  while (p > 1) {
    int q = p - 1;
    Scalar tr = hess_form(p, p) + hess_form(q, q);
    Scalar det =
        (hess_form(q, q) * hess_form(p, p) - hess_form(q, p) * hess_form(p, q));
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M(3, 1);
    M << hess_form(0, 0) * hess_form(0, 0) + hess_form(0, 1) * hess_form(1, 0) -
             tr * hess_form(0, 0) + det,
        hess_form(1, 0) * (hess_form(0, 0) + hess_form(1, 1) - tr),
        hess_form(1, 0) * hess_form(2, 1);
    for (int k = -1; k <= p - 3; ++k) {
      Householder_reflector::Reflector m_ref(M, 0, 0);
      if (k >= 0) {
        m_ref.reflect_left(hess_form, k + 1, k + 3, k, order - 1);
      } else {
        m_ref.reflect_left(hess_form, k + 1, k + 3, 0, order - 1);
      }
      if (k + 4 < p) {
        m_ref.reflect_right(hess_form, 0, k + 4, k + 1, k + 3);
      } else {
        m_ref.reflect_right(hess_form, 0, p, k + 1, k + 3);
      }
      if (find_schur_vectors) {
          m_ref.reflect_right(schur_vectors, 0, order - 1, k + 1, k + 3);
      }
      M(0, 0) = hess_form(k + 2, k + 1);
      M(1, 0) = hess_form(k + 3, k + 1);
      if (k < p - 3) {
        M(2, 0) = hess_form(k + 4, k + 1);
      }
    }
    Householder_reflector::Reflector m_ref(M, 0, 1, 0);
    m_ref.reflect_left(hess_form, q, q + 1, p - 2, order - 1);
    m_ref.reflect_right(hess_form, 0, p, p - 1, p);
    if (find_schur_vectors) {
      m_ref.reflect_right(schur_vectors, 0, order - 1, p - 1, p);
    }
    if (std::abs(hess_form(p, q)) <
        eps * (std::abs(hess_form(q, q)) + std::abs(hess_form(p, p)))) {
      hess_form(p, q) = {};
      --p;
      q = p - 1;
    } else if (std::abs(hess_form(p - 1, q - 1)) <
               eps * (std::abs(hess_form(q - 1, q - 1)) +
                      std::abs(hess_form(q, q)))) {
      hess_form(p - 1, q - 1) = {};
      p -= 2;
      q = p - 1;
    }
  }
  schur_vectors = reduction.get_conversion() * schur_vectors;
  triang = hess_form;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Double_shift<Scalar>::get_triang() const {
  return triang;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
Double_shift<Scalar>::get_schur_vectors() const {
  return schur_vectors;
}

};  // namespace Double_shift
