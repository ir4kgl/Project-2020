#include "QR_shifts.h"

namespace QR_algorithm {

template <typename Scalar>
class QR_algorithm {
 public:
  QR_algorithm(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start,
               Scalar eps, bool find_vectors = false);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_triang()const;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& get_vectors() const;

 private:
  bool symmetric_case = false;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> triang;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> vectors;
};

template <typename Scalar>
QR_algorithm<Scalar>::QR_algorithm(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> start, Scalar eps,
    bool find_vectors) {
  if ((start - start.transpose()).norm() < eps) {
    symmetric_case = true;
    Single_shift::Single_shift decomposition(start, eps, find_vectors);
    triang = decomposition.get_eigenvalues().asDiagonal();
    if (find_vectors) {
      vectors = decomposition.get_eigenvectors();
    }
  } else {
    Double_shift::Double_shift decomposition(start, eps, find_vectors);
    triang = decomposition.get_triang();
    if (find_vectors) {
      vectors = decomposition.get_schur_vectors();
    }
  }
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
QR_algorithm<Scalar>::get_triang() const {
  return triang;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>&
QR_algorithm<Scalar>::get_vectors() const {
  return vectors;
}

};  // namespace QR_algorithm
