#include <Eigen/Dense>

namespace Householder_reflector {

template <typename Scalar>
class Reflector {
 public:
  Reflector(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start,
            int row_ind, int col_ind);
  Reflector(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start,
            int row_ind_ff, int row_ind_ss, int col_ind);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& get_u_vector() const;
  void reflect_left(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start, int row_ff,
      int row_ss, int col_ff, int col_ss) const;
  void reflect_right(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start, int row_ff,
      int row_ss, int col_ff, int col_ss) const;

 private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_vector;
};

template <typename Scalar>
Reflector<Scalar>::Reflector(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start,
    int row_ind, int col_ind) {
  int order = start.rows(), row_cnt = order - row_ind;
  Scalar rho = (start(row_ind, col_ind) > 0) ? -1 : 1;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start_subm =
      start(Eigen::seq(row_ind, order - 1), Eigen::seq(col_ind, col_ind));
  u_vector = start_subm -
             rho * start_subm.norm() *
                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Identity(row_cnt, 1);
  u_vector.normalize();
}

template <typename Scalar>
Reflector<Scalar>::Reflector(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start,
    int row_ind_ff, int row_ind_ss, int col_ind) {
  int row_cnt = row_ind_ss - row_ind_ff + 1;
  Scalar rho = (start(row_ind_ff, col_ind) > 0) ? -1 : 1;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start_subm =
      start(Eigen::seq(row_ind_ff, row_ind_ss), Eigen::seq(col_ind, col_ind));
  u_vector = start_subm -
             rho * start_subm.norm() *
                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Identity(row_cnt, 1);
  u_vector.normalize();
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&
Reflector<Scalar>::get_u_vector() const {
  return u_vector;
}

template <typename Scalar>
void Reflector<Scalar>::reflect_left(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start, int row_ff,
    int row_ss, int col_ff, int col_ss) const {
  start(Eigen::seq(row_ff, row_ss), Eigen::seq(col_ff, col_ss)) -=
      2 * u_vector *
      (u_vector.transpose() *
       start(Eigen::seq(row_ff, row_ss), Eigen::seq(col_ff, col_ss)));
}

template <typename Scalar>
void Reflector<Scalar>::reflect_right(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& start, int row_ff,
    int row_ss, int col_ff, int col_ss) const {
  start(Eigen::seq(row_ff, row_ss), Eigen::seq(col_ff, col_ss)) -=
      2 *
      (start(Eigen::seq(row_ff, row_ss), Eigen::seq(col_ff, col_ss)) *
       u_vector) *
      u_vector.transpose();
}

};  // namespace Householder_reflector
