use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, LitStr, Ident, Token};


/// ----------------------------------------------------------------------
///                       Input Parsing and Helper Functions
/// ----------------------------------------------------------------------
///
/// The input parser remains as `PyOperatorInput`. It expects a single identifier
/// (the inner type) since each macro is tied to a fixed operator type.
struct PyOperatorInput {
    inner: Ident,
}

impl Parse for PyOperatorInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner: Ident = input.parse()?;
        Ok(PyOperatorInput { inner })
    }
}

/// A common helper function to generate the wrapper struct and PyO3 attributes.
/// It returns a tuple containing:
/// - The new wrapper identifier (formed as "Py" + inner type name)
/// - A literal for the inner type's name (to be used in the `#[pyclass(name = "...")]` attribute)
///
/// # Example
///
/// For an inner operator named `BitFlipMutation`, this function generates:
/// - A wrapper identifier `PyBitFlipMutation`
/// - A literal `"BitFlipMutation"`
fn generate_wrapper(inner: &Ident) -> (Ident, LitStr) {
    let span = inner.span();
    let wrapper_name = format!("Py{}", inner);
    let wrapper_ident = Ident::new(&wrapper_name, span);
    let inner_name_lit = LitStr::new(&inner.to_string(), span);
    (wrapper_ident, inner_name_lit)
}

/// ----------------------------------------------------------------------
///                   Mutation Operator Macro
/// ----------------------------------------------------------------------
///
/// Generates a Python wrapper for a mutation operator.
/// (The following code remains unchanged.)
fn generate_py_operator_mutation(inner: Ident) -> proc_macro2::TokenStream {
    let (wrapper_ident, inner_name_lit) = generate_wrapper(&inner);
    // Define the mutation-specific method.
    let operator_method = quote! {
        #[pyo3(signature = (population, seed=None))]
        pub fn mutate<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            population: numpy::PyReadonlyArrayDyn<'py, f64>,
            seed: Option<u64>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
            let owned_population = population.to_owned_array();
            let mut owned_population = owned_population.into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Population numpy array must be 2D."))?;
            let mut rng = moors::random::MOORandomGenerator::new_from_seed(seed);
            self.inner.operate(&mut owned_population, 1.0, &mut rng);
            Ok(numpy::ToPyArray::to_pyarray(&owned_population, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Clone, Debug)]
        pub struct #wrapper_ident {
            pub inner: #inner,
        }

        #[pyo3::prelude::pymethods]
        impl #wrapper_ident {
            #operator_method
        }
    }
}

#[proc_macro]
pub fn py_operator_mutation(input: TokenStream) -> TokenStream {
    let PyOperatorInput { inner } = parse_macro_input!(input as PyOperatorInput);
    generate_py_operator_mutation(inner).into()
}

/// ----------------------------------------------------------------------
///                   Crossover Operator Macro
/// ----------------------------------------------------------------------
///
/// Generates a Python wrapper for a crossover operator.
fn generate_py_operator_crossover(inner: Ident) -> proc_macro2::TokenStream {
    let (wrapper_ident, inner_name_lit) = generate_wrapper(&inner);
    // Define the crossover-specific method.
    let operator_method = quote! {
        #[pyo3(signature = (parents_a, parents_b, seed=None))]
        pub fn crossover<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            parents_a: numpy::PyReadonlyArrayDyn<'py, f64>,
            parents_b: numpy::PyReadonlyArrayDyn<'py, f64>,
            seed: Option<u64>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
            let owned_parents_a = parents_a.to_owned_array();
            let owned_parents_b = parents_b.to_owned_array();
            let owned_parents_a = owned_parents_a.into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("parent_a numpy array must be 2D."))?;
            let owned_parents_b = owned_parents_b.into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("parent_b numpy array must be 2D."))?;
            let mut rng = moors::random::MOORandomGenerator::new_from_seed(seed);
            let offspring = self.inner.operate(&owned_parents_a, &owned_parents_b, 1.0, &mut rng);
            Ok(numpy::ToPyArray::to_pyarray(&offspring, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Clone, Debug)]
        pub struct #wrapper_ident {
            pub inner: #inner,
        }

        #[pyo3::prelude::pymethods]
        impl #wrapper_ident {
            #operator_method
        }
    }
}

#[proc_macro]
pub fn py_operator_crossover(input: TokenStream) -> TokenStream {
    let PyOperatorInput { inner } = parse_macro_input!(input as PyOperatorInput);
    generate_py_operator_crossover(inner).into()
}

/// ----------------------------------------------------------------------
///                   Sampling Operator Macro
/// ----------------------------------------------------------------------
///
/// Generates a Python wrapper for a sampling operator.
fn generate_py_operator_sampling(inner: Ident) -> proc_macro2::TokenStream {
    let (wrapper_ident, inner_name_lit) = generate_wrapper(&inner);
    // Define the sampling-specific method.
    let operator_method = quote! {
        #[pyo3(signature = (population_size, n_vars, seed=None))]
        pub fn sample<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            population_size: usize,
            n_vars: usize,
            seed: Option<u64>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
            let mut rng = moors::random::MOORandomGenerator::new_from_seed(seed);
            let sampled_population = self.inner.operate(population_size, n_vars, &mut rng);
            Ok(numpy::ToPyArray::to_pyarray(&sampled_population, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Clone, Debug)]
        pub struct #wrapper_ident {
            pub inner: #inner,
        }

        #[pyo3::prelude::pymethods]
        impl #wrapper_ident {
            #operator_method
        }
    }
}

#[proc_macro]
pub fn py_operator_sampling(input: TokenStream) -> TokenStream {
    let PyOperatorInput { inner } = parse_macro_input!(input as PyOperatorInput);
    generate_py_operator_sampling(inner).into()
}

/// ----------------------------------------------------------------------
///                   Duplicates Operator Macro
/// ----------------------------------------------------------------------
///
/// Generates a Python wrapper for a duplicates operator (population cleaner).
fn generate_py_operator_duplicates(inner: Ident) -> proc_macro2::TokenStream {
    let (wrapper_ident, inner_name_lit) = generate_wrapper(&inner);
    // Define the duplicates-specific method.
    let operator_method = quote! {
        #[pyo3(signature = (population, reference=None))]
        pub fn remove_duplicates<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            population: numpy::PyReadonlyArrayDyn<'py, f64>,
            reference: Option<numpy::PyReadonlyArrayDyn<'py, f64>>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
            let population = population.to_owned_array();
            let population = population.into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Population numpy array must be 2D."))?;
            let reference = reference
                .map(|ref_arr| {
                    ref_arr.to_owned_array().into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Reference numpy array must be 2D."))
                })
                .transpose()?;
            let clean_population = self.inner.remove(&population, reference.as_ref());
            Ok(numpy::ToPyArray::to_pyarray(&clean_population, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Clone, Debug)]
        pub struct #wrapper_ident {
            pub inner: #inner,
        }

        #[pyo3::prelude::pymethods]
        impl #wrapper_ident {
            #operator_method
        }
    }
}

#[proc_macro]
pub fn py_operator_duplicates(input: TokenStream) -> TokenStream {
    let PyOperatorInput { inner } = parse_macro_input!(input as PyOperatorInput);
    generate_py_operator_duplicates(inner).into()
}

/// ----------------------------------------------------------------------
///           Parser for a Comma-Separated List of Operators
/// ----------------------------------------------------------------------
///
/// This parser is used to parse a comma-separated list of operator identifiers.
struct OpsList {
    ops: Punctuated<Ident, Token![,]>,
}

impl Parse for OpsList {
    fn parse(input: ParseStream) -> Result<Self> {
        let ops = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        Ok(OpsList { ops })
    }
}

/// ----------------------------------------------------------------------
///         Registration Macro for Mutation Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// This macro registers a list of mutation operators and does the following:
/// 1. Invokes the individual `py_operator_mutation!` macro for each operator.
/// 2. Generates an enum named `MutationOperatorDispatcher` with variants for each operator.
/// 3. Generates `From<T>` implementations so that each concrete operator converts automatically.
/// 4. Implements the trait `MutationOperator` for the enum by delegating method calls,
///    as well as `GeneticOperator` by returning the name.
/// 5. Generates the `unwrap_mutation_operator` function that extracts the operator from a PyObject
///    and converts it to the enum.
///
/// # Example
///
/// ```rust
/// register_py_operators_mutation!(BitFlipMutation, ScrambleMutation);
/// ```
#[proc_macro]
pub fn register_py_operators_mutation(input: TokenStream) -> TokenStream {
    let OpsList { ops } = parse_macro_input!(input as OpsList);

    // Generate enum variants for each operator.
    let enum_variants = ops.iter().map(|op| {
        quote! {
            #op(#op)
        }
    });
    let enum_def = quote! {
        #[derive(Clone, Debug)]
        pub enum MutationOperatorDispatcher {
            #(#enum_variants),*
        }
    };

    // Generate From implementations for each concrete type.
    let from_impls = ops.iter().map(|op| {
        quote! {
            impl From<#op> for MutationOperatorDispatcher {
                fn from(operator: #op) -> Self {
                    MutationOperatorDispatcher::#op(operator)
                }
            }
        }
    });

    // Generate implementation for MutationOperator trait.
    let mutate_arms = ops.iter().map(|op| {
        quote! {
            MutationOperatorDispatcher::#op(inner) => inner.mutate(individual, rng),
        }
    });
    let mutation_impl = quote! {
        impl moors::operators::MutationOperator for MutationOperatorDispatcher {
            fn mutate<'a>(
                &self,
                individual: moors::genetic::IndividualGenesMut<'a>,
                rng: &mut dyn moors::random::RandomGenerator,
            ) {
                match self {
                    #(#mutate_arms)*
                }
            }
        }
    };

    // Generate implementation for GeneticOperator trait.
    let name_arms = ops.iter().map(|op| {
        quote! {
            MutationOperatorDispatcher::#op(inner) => inner.name(),
        }
    });
    let genetic_impl = quote! {
        impl moors::operators::GeneticOperator for MutationOperatorDispatcher {
            fn name(&self) -> String {
                match self {
                    #(#name_arms)*
                }
            }
        }
    };

    // Generate calls to each individual mutation operator macro.
    let macro_calls = {
        let calls = ops.iter().map(|op| {
            quote! {
                pymoors_macros::py_operator_mutation!(#op);
            }
        });
        quote! { #(#calls)* }
    };

    // Generate extraction arms for each operator.
    let extract_arms = ops.iter().map(|op| {
        let op_str = op.to_string();
        let wrapper = Ident::new(&format!("Py{}", op_str), op.span());
        quote! {
            if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                return Ok(MutationOperatorDispatcher::from(extracted.inner));
            }
        }
    });

    let unwrap_fn = quote! {
        pub fn unwrap_mutation_operator(py_obj: pyo3::PyObject) -> pyo3::PyResult<MutationOperatorDispatcher> {
            pyo3::Python::with_gil(|py| {
                #(#extract_arms)*
                Err(pyo3::exceptions::PyValueError::new_err("Could not extract a valid mutation operator"))
            })
        }
    };

    let expanded = quote! {
        #enum_def
        #(#from_impls)*
        #mutation_impl
        #genetic_impl
        #macro_calls
        #unwrap_fn
    };

    TokenStream::from(expanded)
}

/// ----------------------------------------------------------------------
///         Registration Macro for Crossover Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Generates an enum named `CrossoverOperatorDispatcher` with From implementations,
/// implements the traits `CrossoverOperator` and `GeneticOperator`, invokes the
/// corresponding PyO3 wrappers, and creates an unwrap function.
#[proc_macro]
pub fn register_py_operators_crossover(input: TokenStream) -> TokenStream {
    let OpsList { ops } = parse_macro_input!(input as OpsList);

    let enum_variants = ops.iter().map(|op| {
        quote! {
            #op(#op)
        }
    });
    let enum_def = quote! {
        #[derive(Clone, Debug)]
        pub enum CrossoverOperatorDispatcher {
            #(#enum_variants),*
        }
    };

    let from_impls = ops.iter().map(|op| {
        quote! {
            impl From<#op> for CrossoverOperatorDispatcher {
                fn from(operator: #op) -> Self {
                    CrossoverOperatorDispatcher::#op(operator)
                }
            }
        }
    });

    // Implement CrossoverOperator trait.
    let crossover_arms = ops.iter().map(|op| {
        quote! {
            CrossoverOperatorDispatcher::#op(inner) => inner.crossover(parent_a, parent_b, rng),
        }
    });
    let crossover_impl = quote! {
        impl moors::operators::CrossoverOperator for CrossoverOperatorDispatcher {
            fn crossover(
                &self,
                parent_a: &moors::genetic::IndividualGenes,
                parent_b: &moors::genetic::IndividualGenes,
                rng: &mut dyn moors::random::RandomGenerator,
            ) -> (moors::genetic::IndividualGenes, moors::genetic::IndividualGenes) {
                match self {
                    #(#crossover_arms)*
                }
            }
        }
    };

    // Implement GeneticOperator trait.
    let name_arms = ops.iter().map(|op| {
        quote! {
            CrossoverOperatorDispatcher::#op(inner) => inner.name(),
        }
    });
    let genetic_impl = quote! {
        impl moors::operators::GeneticOperator for CrossoverOperatorDispatcher {
            fn name(&self) -> String {
                match self {
                    #(#name_arms)*
                }
            }
        }
    };

    let macro_calls = {
        let calls = ops.iter().map(|op| {
            quote! {
                pymoors_macros::py_operator_crossover!(#op);
            }
        });
        quote! { #(#calls)* }
    };

    let extract_arms = ops.iter().map(|op| {
        let op_str = op.to_string();
        let wrapper = Ident::new(&format!("Py{}", op_str), op.span());
        quote! {
            if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                return Ok(CrossoverOperatorDispatcher::from(extracted.inner));
            }
        }
    });

    let unwrap_fn = quote! {
        pub fn unwrap_crossover_operator(py_obj: pyo3::PyObject) -> pyo3::PyResult<CrossoverOperatorDispatcher> {
            pyo3::Python::with_gil(|py| {
                #(#extract_arms)*
                Err(pyo3::exceptions::PyValueError::new_err("Could not extract a valid crossover operator"))
            })
        }
    };

    let expanded = quote! {
        #enum_def
        #(#from_impls)*
        #crossover_impl
        #genetic_impl
        #macro_calls
        #unwrap_fn
    };

    TokenStream::from(expanded)
}

/// ----------------------------------------------------------------------
///         Registration Macro for Sampling Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Generates an enum named `SamplingOperatorDispatcher` with From implementations,
/// implements the traits `SamplingOperator` and `GeneticOperator`, invokes the
/// corresponding PyO3 wrappers, and creates an unwrap function.
#[proc_macro]
pub fn register_py_operators_sampling(input: TokenStream) -> TokenStream {
    let OpsList { ops } = parse_macro_input!(input as OpsList);

    let enum_variants = ops.iter().map(|op| {
        quote! {
            #op(#op)
        }
    });
    let enum_def = quote! {
        #[derive(Clone, Debug)]
        pub enum SamplingOperatorDispatcher {
            #(#enum_variants),*
        }
    };

    let from_impls = ops.iter().map(|op| {
        quote! {
            impl From<#op> for SamplingOperatorDispatcher {
                fn from(operator: #op) -> Self {
                    SamplingOperatorDispatcher::#op(operator)
                }
            }
        }
    });

    // Implement SamplingOperator trait.
    let sample_arms = ops.iter().map(|op| {
        quote! {
            SamplingOperatorDispatcher::#op(inner) => inner.sample_individual(n_vars, rng),
        }
    });
    let sampling_impl = quote! {
        impl moors::operators::SamplingOperator for SamplingOperatorDispatcher {
            fn sample_individual(&self, n_vars: usize, rng: &mut dyn moors::random::RandomGenerator) -> moors::genetic::IndividualGenes {
                match self {
                    #(#sample_arms)*
                }
            }
        }
    };

    // Implement GeneticOperator trait.
    let name_arms = ops.iter().map(|op| {
        quote! {
            SamplingOperatorDispatcher::#op(inner) => inner.name(),
        }
    });
    let genetic_impl = quote! {
        impl moors::operators::GeneticOperator for SamplingOperatorDispatcher {
            fn name(&self) -> String {
                match self {
                    #(#name_arms)*
                }
            }
        }
    };

    let macro_calls = {
        let calls = ops.iter().map(|op| {
            quote! {
                pymoors_macros::py_operator_sampling!(#op);
            }
        });
        quote! { #(#calls)* }
    };

    let extract_arms = ops.iter().map(|op| {
        let op_str = op.to_string();
        let wrapper = Ident::new(&format!("Py{}", op_str), op.span());
        quote! {
            if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                return Ok(SamplingOperatorDispatcher::from(extracted.inner));
            }
        }
    });

    let unwrap_fn = quote! {
        pub fn unwrap_sampling_operator(py_obj: pyo3::PyObject) -> pyo3::PyResult<SamplingOperatorDispatcher> {
            pyo3::Python::with_gil(|py| {
                #(#extract_arms)*
                Err(pyo3::exceptions::PyValueError::new_err("Could not extract a valid sampling operator"))
            })
        }
    };

    let expanded = quote! {
        #enum_def
        #(#from_impls)*
        #sampling_impl
        #genetic_impl
        #macro_calls
        #unwrap_fn
    };

    TokenStream::from(expanded)
}

/// ----------------------------------------------------------------------
///         Registration Macro for Duplicates Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Generates an enum named `DuplicatesCleanerDispatcher` with From implementations,
/// implements the trait `PopulationCleaner`, invokes the corresponding PyO3 wrappers,
/// and creates an unwrap function.
#[proc_macro]
pub fn register_py_operators_duplicates(input: TokenStream) -> TokenStream {
    let OpsList { ops } = parse_macro_input!(input as OpsList);

    let enum_variants = ops.iter().map(|op| {
        quote! {
            #op(#op)
        }
    });
    let enum_def = quote! {
        #[derive(Clone, Debug)]
        pub enum DuplicatesCleanerDispatcher {
            #(#enum_variants),*
        }
    };

    let from_impls = ops.iter().map(|op| {
        quote! {
            impl From<#op> for DuplicatesCleanerDispatcher {
                fn from(operator: #op) -> Self {
                    DuplicatesCleanerDispatcher::#op(operator)
                }
            }
        }
    });

    // Implement PopulationCleaner trait.
    let remove_arms = ops.iter().map(|op| {
        quote! {
            DuplicatesCleanerDispatcher::#op(inner) => inner.remove(population, reference),
        }
    });
    let cleaner_impl = quote! {
        impl moors::duplicates::PopulationCleaner for DuplicatesCleanerDispatcher {
            fn remove(
                &self,
                population: &moors::genetic::PopulationGenes,
                reference: Option<&moors::genetic::PopulationGenes>,
            ) -> moors::genetic::PopulationGenes {
                match self {
                    #(#remove_arms)*
                }
            }
        }
    };

    let macro_calls = {
        let calls = ops.iter().map(|op| {
            quote! {
                pymoors_macros::py_operator_duplicates!(#op);
            }
        });
        quote! { #(#calls)* }
    };

    let extract_arms = ops.iter().map(|op| {
        let op_str = op.to_string();
        let wrapper = Ident::new(&format!("Py{}", op_str), op.span());
        quote! {
            if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                return Ok(DuplicatesCleanerDispatcher::from(extracted.inner));
            }
        }
    });

    let unwrap_fn = quote! {
        pub fn unwrap_duplicates_operator(py_obj: pyo3::PyObject) -> pyo3::PyResult<DuplicatesCleanerDispatcher> {
            pyo3::Python::with_gil(|py| {
                #(#extract_arms)*
                Err(pyo3::exceptions::PyValueError::new_err("Could not extract a valid duplicates operator"))
            })
        }
    };

    let expanded = quote! {
        #enum_def
        #(#from_impls)*
        #cleaner_impl
        #macro_calls
        #unwrap_fn
    };

    TokenStream::from(expanded)
}

/// Implementation for the `py_algorithm` macro.
///
/// This macro receives an identifier of an already defined struct (for example, `PyNsga2`)
/// and generates an implementation block (with `#[pymethods]`) that defines:
///
/// - `run(&mut self) -> PyResult<()>`: calls `self.algorithm.run()` and maps any error.
/// - A getter `population(&self, py: Python) -> PyResult<PyObject>` that converts the
///   algorithm's population data to a Python object.
///
/// # Example
///
/// Assuming you have defined:
///
/// ```rust
/// #[pyclass(name = "Nsga2", unsendable)]
/// pub struct PyNsga2 {
///     pub algorithm: Nsga2,
/// }
/// ```
///
/// You can then invoke the macro as:
///
/// ```rust
/// py_algorithm!(PyNsga2);
/// ```
///
/// and the macro will generate the implementation block for `PyNsga2`.
#[proc_macro]
pub fn py_algorithm_impl(input: TokenStream) -> TokenStream {
    // Parse the input identifier, e.g. "PyNsga2".
    let py_struct_ident = parse_macro_input!(input as Ident);

    let expanded = quote! {
        #[pymethods]
        impl #py_struct_ident {
            /// Calls the underlying algorithm's `run()` method,
            /// converting any error to a Python runtime error.
            pub fn run(&mut self) -> pyo3::PyResult<()> {
                self.algorithm
                    .run()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            }

            /// Getter for the algorithm's population.
            /// It converts the internal population members (genes, fitness, rank, constraints)
            /// to Python objects using NumPy.
            #[getter]
            pub fn population(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
                let schemas_module = py.import("pymoors.schemas")?;
                let population_class = schemas_module.getattr("Population")?;
                let population = &self.algorithm.inner.population;

                let py_genes = population.genes.to_pyarray(py);
                let py_fitness = population.fitness.to_pyarray(py);

                let py_rank = if let Some(ref r) = population.rank {
                    r.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };

                let py_constraints = if let Some(ref c) = population.constraints {
                    c.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };

                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("genes", py_genes)?;
                kwargs.set_item("fitness", py_fitness)?;
                kwargs.set_item("rank", py_rank)?;
                kwargs.set_item("constraints", py_constraints)?;

                let py_instance = population_class.call((), Some(&kwargs))?;
                Ok(py_instance.into_py(py))
            }
        }
    };

    TokenStream::from(expanded)
}
