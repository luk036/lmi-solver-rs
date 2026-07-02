use crate::ldlt_mgr::LDLTMgr;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleFeas, SingleCut};
use ndarray::Array2;

pub type Cut = (Arr, SingleCut);

/// The `LMIOracle` struct represents an oracle for a Linear Matrix Inequality (LMI) constraint.
///
/// A Linear Matrix Inequality has the form:
/// $$ F(x) = F_0 + \sum_{i=1}^{m} x_i F_i \succ 0 $$
///
/// where $$ F_i $$ are symmetric matrices and $$ F(x) $$ must be positive definite.
///
/// It contains the necessary data to evaluate the LMI constraint, including the matrix `mat_f`,
/// the matrix `mat_f0`, and an `LDLTMgr` instance for managing the Cholesky decomposition.
/// This oracle can be used to check the feasibility of a given point with respect to the LMI constraint.
pub struct LMIOracle {
    mat_f: Vec<Array2<f64>>,
    mat_f0: Array2<f64>,
    ldlt_mgr: LDLTMgr,
}

impl LMIOracle {
    /// This function initializes a new `LMIOracle` struct with given matrices and an `LDLTMgr` instance.
    ///
    /// Arguments:
    ///
    /// * `mat_f`: The `mat_f` parameter is a vector of 2D arrays of type `f64`.
    /// * `mat_b`: The `mat_b` parameter is an `Array2<f64>` type, which represents a 2-dimensional array
    ///   of f64 (floating point numbers).
    ///
    /// Returns:
    ///
    /// An instance of the `LMIOracle` struct is being returned.
    pub fn new(mat_f: Vec<Array2<f64>>, mat_b: Array2<f64>) -> Self {
        let ldlt_mgr = LDLTMgr::new(mat_b.nrows());
        LMIOracle {
            mat_f,
            mat_f0: mat_b,
            ldlt_mgr,
        }
    }
}

impl OracleFeas<Arr> for LMIOracle {
    type CutChoice = SingleCut; // single cut

    /// Assesses feasibility of $$ x_c $$ with respect to the LMI constraint.
    ///
    /// $$ F(x_c) = F_0 + \sum_{i=1}^{m} x_i F_i \succ 0 $$
    ///
    /// If $$ F(x_c) $$ is not positive definite, returns a cutting plane
    /// with gradient $$ g_i = w^T F_i w $$ and $$ \epsilon = -D_{kk} $$.
    ///
    /// Arguments:
    ///
    /// * `xc`: The `xc` parameter is a reference to an `Array1<f64>`, which represents a one-dimensional
    ///   array of floating-point numbers. This array is used as input to the function for some calculations
    ///   related to feasibility assessment. The function uses the struct fields `mat_f0` and `mat_f` to
    ///   perform the feasibility check.
    fn assess_feas(&mut self, xc: &Arr) -> Option<Cut> {
        fn get_elem(
            mat_f0: &Array2<f64>,
            mat_f: &[Array2<f64>],
            xc: &[f64],
            i: usize,
            j: usize,
        ) -> f64 {
            mat_f0[(i, j)]
                - mat_f
                    .iter()
                    .zip(xc.iter())
                    .map(|(mat_fk, xk)| mat_fk[(i, j)] * xk)
                    .sum::<f64>()
        }

        let xc_slice = xc.data();
        let get_elem = |i: usize, j: usize| get_elem(&self.mat_f0, &self.mat_f, xc_slice, i, j);

        if self.ldlt_mgr.factor(get_elem) {
            None
        } else {
            let epsilon = self.ldlt_mgr.witness();
            let grad_vec: Vec<f64> = self
                .mat_f
                .iter()
                .map(|mat_fk| self.ldlt_mgr.sym_quad(mat_fk))
                .collect();
            Some((Arr::from(grad_vec), SingleCut(epsilon)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
    use ellalgo_rs::ell::Ell;
    use ndarray::ShapeError;

    struct MyOracle {
        c: Arr,
        lmi1: LMIOracle,
        lmi2: LMIOracle,
    }

    impl OracleOptim<Arr> for MyOracle {
        type CutChoice = SingleCut; // single cut

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            if let Some(cut) = self.lmi1.assess_feas(xc) {
                return (cut, false);
            }

            if let Some(cut) = self.lmi2.assess_feas(xc) {
                return (cut, false);
            }

            let f0 = self.c.dot(xc);
            let func_val = f0 - *gamma;
            if func_val > 0.0 {
                return ((self.c.clone(), SingleCut(func_val)), false);
            }

            *gamma = f0;
            ((self.c.clone(), SingleCut(0.0)), true)
        }
    }

    fn run_lmi(oracle1: LMIOracle, oracle2: LMIOracle) -> usize {
        let xinit = Arr::new(3);
        let mut ellip = Ell::new_with_scalar(10.0, xinit);
        let mut omega = MyOracle {
            c: Arr::from(vec![1.0, -1.0, 1.0]),
            lmi1: oracle1,
            lmi2: oracle2,
        };
        let mut gamma = f64::INFINITY;
        let (xbest, num_iters) =
            cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &Options::default());
        assert!(xbest.is_some());
        num_iters
    }

    #[test]
    fn test_lmi() -> Result<(), ShapeError> {
        let f1 = vec![
            Array2::from_shape_vec((2, 2), vec![-7.0, -11.0, -11.0, 3.0])?,
            Array2::from_shape_vec((2, 2), vec![7.0, -18.0, -18.0, 8.0])?,
            Array2::from_shape_vec((2, 2), vec![-2.0, -8.0, -8.0, 1.0])?,
        ];
        let b1 = Array2::from_shape_vec((2, 2), vec![33.0, -9.0, -9.0, 26.0])?;
        let f2 = vec![
            Array2::from_shape_vec(
                (3, 3),
                vec![-21.0, -11.0, 0.0, -11.0, 10.0, 8.0, 0.0, 8.0, 5.0],
            )?,
            Array2::from_shape_vec(
                (3, 3),
                vec![0.0, 10.0, 16.0, 10.0, -10.0, -10.0, 16.0, -10.0, 3.0],
            )?,
            Array2::from_shape_vec(
                (3, 3),
                vec![-5.0, 2.0, -17.0, 2.0, -6.0, 8.0, -17.0, 8.0, 6.0],
            )?,
        ];
        let b2 = Array2::from_shape_vec(
            (3, 3),
            vec![14.0, 9.0, 40.0, 9.0, 91.0, 10.0, 40.0, 10.0, 15.0],
        )?;

        let oracle1 = LMIOracle::new(f1, b1);
        let oracle2 = LMIOracle::new(f2, b2);
        let result = run_lmi(oracle1, oracle2);
        assert_eq!(result, 281);
        Ok(())
    }

    #[test]
    fn test_lmi_oracle() {
        let f1 = vec![
            Array2::from_shape_vec((2, 2), vec![-7.0, -11.0, -11.0, 3.0]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![7.0, -18.0, -18.0, 8.0]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![-2.0, -8.0, -8.0, 1.0]).unwrap(),
        ];
        let b1 = Array2::from_shape_vec((2, 2), vec![33.0, -9.0, -9.0, 26.0]).unwrap();
        let mut oracle = LMIOracle::new(f1, b1);
        let x_vec = Arr::from(vec![1.0, 1.0, 1.0]);
        let res = oracle.assess_feas(&x_vec);
        assert!(res.is_some());
        let x_vec2 = Arr::from(vec![0.0, 0.0, 0.0]);
        let res2 = oracle.assess_feas(&x_vec2);
        assert!(res2.is_none());
    }
}
