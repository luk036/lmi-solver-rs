//! # lmi-solver-rs
//!
//! LMI (Linear Matrix Inequality) solver using the Ellipsoid Method.
//!
//! This crate provides an implementation of an LMI feasibility oracle that
//! can be used with the ellipsoid method from [ellalgo-rs](https://github.com/luk036/ellalgo-rs)
//! to solve optimization problems with Linear Matrix Inequality constraints.
//!
//! ## Modules
//!
//! - [`ldlt_mgr`] - LDL^T factorization manager for positive definiteness checking
//! - [`lmi_oracle`] - Linear Matrix Inequality (LMI) feasibility oracle
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ellalgo_rs::arr::Arr;
//! use ellalgo_rs::cutting_plane::OracleFeas;
//! use lmi_solver_rs::lmi_oracle::LMIOracle;
//! use ndarray::Array2;
//!
//! let f = vec![
//!     Array2::from_shape_vec((2, 2), vec![-7.0, -11.0, -11.0, 3.0]).unwrap(),
//!     Array2::from_shape_vec((2, 2), vec![7.0, -18.0, -18.0, 8.0]).unwrap(),
//!     Array2::from_shape_vec((2, 2), vec![-2.0, -8.0, -8.0, 1.0]).unwrap(),
//! ];
//! let b = Array2::from_shape_vec((2, 2), vec![33.0, -9.0, -9.0, 26.0]).unwrap();
//! let mut oracle = LMIOracle::new(f, b);
//! let x = Arr::from(vec![1.0, 1.0, 1.0]);
//! let cut = oracle.assess_feas(&x);
//! assert!(cut.is_some());
//! ```

pub mod ldlt_mgr;
pub mod lmi_oracle;
