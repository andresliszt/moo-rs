use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::{Fields, Ident, ItemEnum, LitStr, Type, parse_macro_input};

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
        pub fn operate<'py>(
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
        #[derive(Debug, Clone)]
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
        pub fn operate<'py>(
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
        #[derive(Debug, Clone)]
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
        #[pyo3(signature = (population_size, num_vars, seed=None))]
        pub fn operate<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            population_size: usize,
            num_vars: usize,
            seed: Option<u64>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
            let mut rng = moors::random::MOORandomGenerator::new_from_seed(seed);
            let sampled_population = self.inner.operate(population_size, num_vars, &mut rng);
            Ok(numpy::ToPyArray::to_pyarray(&sampled_population, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Debug, Clone)]
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
            let clean_population = self.inner.remove(population, reference.as_ref());
            Ok(numpy::ToPyArray::to_pyarray(&clean_population, py))
        }
    };

    quote! {
        #[pyo3::prelude::pyclass(name = #inner_name_lit)]
        #[derive(Debug, Clone)]
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
///         Registration Macro for Mutation Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Applies to an enum whose variants are all tuple‐variants `Variant(Type)`.
/// For each variant this attribute will:
/// - Generate `impl From<Type> for MutationOperatorDispatcher`
/// - Implement `moors::operators::MutationOperator` by delegating `mutate(...)`
/// - Emit `py_operator_mutation!(Type)` for each Rust‐native operator
/// - Add a constructor
///     `fn from_python_operator(py_obj: PyObject) -> PyResult<Self>`
///   that extracts the correct variant from a Python object.
///
/// Note: this macro will also honor a variant named
/// `CustomPyMutationOperatorWrapper(CustomPyMutationOperatorWrapper)`,
/// but will skip emitting `py_operator_mutation!` for it.
#[proc_macro_attribute]
pub fn register_py_operators_mutation(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the enum the user wrote
    let input_enum: ItemEnum = parse_macro_input!(item as ItemEnum);
    let enum_ident = &input_enum.ident;

    // Collect every variant, assuming each is `Variant(Type)`
    let ops: Vec<(proc_macro2::Ident, Type)> = input_enum
        .variants
        .iter()
        .map(|v| {
            let ty = match &v.fields {
                Fields::Unnamed(f) if f.unnamed.len() == 1 => f.unnamed[0].ty.clone(),
                other => panic!("Expected tuple‐variant with one field, got {:?}", other),
            };
            (v.ident.clone(), ty)
        })
        .collect();

    // impl From<Type> for each variant
    let from_impls = ops.iter().map(|(var, ty)| {
        quote! {
            impl From<#ty> for #enum_ident {
                fn from(op: #ty) -> Self { #enum_ident::#var(op) }
            }
        }
    });

    // MutationOperator impl
    let mutate_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.mutate(individual, rng),
        }
    });
    let operate_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.operate(population, mutation_rate, rng),
        }
    });

    let mutation_impl = quote! {
        impl moors::operators::MutationOperator for #enum_ident {
            fn mutate<'a>(
                &self,
                individual: ndarray::ArrayViewMut1<'a, f64>,
                rng: &mut impl moors::random::RandomGenerator,
            ) {
                match self { #(#mutate_match)* }
            }
            fn operate(
                &self,
                population: &mut ndarray::Array2<f64>,
                mutation_rate: f64,
                rng: &mut impl moors::random::RandomGenerator,
            ) {
                match self { #(#operate_match)* }
            }
        }
    };

    // Emit py_operator_mutation!(Type) for each operator except the custom wrapper
    let macro_calls = ops.iter().filter_map(|(var, ty)| {
        if var == "CustomPyMutationOperatorWrapper" {
            None
        } else {
            Some(quote! { pymoors_macros::py_operator_mutation!(#ty); })
        }
    });
    // from_python_operator constructor: try the PyMutation wrappers first…
    let mut extract_arms = Vec::new();
    for (var, _ty) in &ops {
        if var != "CustomPyMutationOperatorWrapper" {
            let wrapper = format_ident!("Py{}", var);
            extract_arms.push(quote! {
                if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                    return Ok(#enum_ident::from(extracted.inner));
                }
            });
        }
    }
    // …and only if none of those matched, try the custom wrapper itself
    extract_arms.push(quote! {
        if let Ok(extracted) = py_obj.extract::<CustomPyMutationOperatorWrapper>(py) {
            return Ok(#enum_ident::from(extracted));
        }
    });

    let ctor_impl = quote! {
        impl #enum_ident {
            /// Convert a Python‐side operator into this dispatcher.
            pub fn from_python_operator(
                py_obj: pyo3::PyObject
            ) -> pyo3::PyResult<Self> {
                pyo3::Python::with_gil(|py| {
                    #(#extract_arms)*
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "Could not extract a valid mutation operator",
                    ))
                })
            }
        }
    };

    // Emit the enum plus all generated code
    TokenStream::from(quote! {
        #input_enum
        #(#from_impls)*
        #mutation_impl
        #(#macro_calls)*
        #ctor_impl
    })
}

/// ----------------------------------------------------------------------
///         Registration Macro for Crossover Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Applies to an enum whose variants are of the form `Variant(Type)`. For each
/// variant this attribute will:
/// - Generate `impl From<Type> for CrossoverEnumDispatcher`
/// - Implement `moors::operators::CrossoverOperator` by delegating `crossover(...)`
/// - Emit a call to `py_operator_crossover!(Type)` so the Python wrapper is registered
/// - Add an associated constructor:
///     `fn from_python_operator(py_obj: PyObject) -> PyResult<Self>`
///   which extracts the correct variant from a `PyObject`.
#[proc_macro_attribute]
pub fn register_py_operators_crossover(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the enum the user provided
    let input_enum: ItemEnum = parse_macro_input!(item as ItemEnum);
    let enum_ident = &input_enum.ident;

    // Collect (VariantIdent, FieldType) for each tuple‐variant `Variant(Type)`
    let ops: Vec<(proc_macro2::Ident, Type)> = input_enum
        .variants
        .iter()
        .filter_map(|v| match &v.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => {
                Some((v.ident.clone(), f.unnamed[0].ty.clone()))
            }
            _ => None, // ignore unit or struct‐like variants
        })
        .collect();

    // impl From<T> for enum
    let from_impls = ops.iter().map(|(var, ty)| {
        quote! {
            impl From<#ty> for #enum_ident {
                fn from(op: #ty) -> Self {
                    #enum_ident::#var(op)
                }
            }
        }
    });

    // impl CrossoverOperator by delegating to each variant
    let crossover_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.crossover(parent_a, parent_b, rng),
        }
    });
    let operate_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.operate(parents_a, parents_b, crossover_rate, rng),
        }
    });
    let crossover_impl = quote! {
        impl moors::operators::CrossoverOperator for #enum_ident {
            fn crossover(
                &self,
                parent_a: &ndarray::Array1<f64>,
                parent_b: &ndarray::Array1<f64>,
                rng: &mut impl moors::random::RandomGenerator,
            ) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
                match self { #(#crossover_match)* }
            }
            fn operate(
                &self,
                parents_a: &ndarray::Array2<f64>,
                parents_b: &ndarray::Array2<f64>,
                crossover_rate: f64,
                rng: &mut impl moors::random::RandomGenerator,
            ) -> ndarray::Array2<f64> {
                match self { #(#operate_match)* }
            }
        }
    };

    // invoke py_operator_crossover!(Type) for each Rust operator type
    let macro_calls = ops.iter().filter_map(|(var, ty)| {
        if var == "CustomPyCrossoverOperatorWrapper" {
            None
        } else {
            Some(quote! { pymoors_macros::py_operator_crossover!(#ty); })
        }
    });
    // from_python_operator constructor: try the PyCrossover wrappers first…
    let mut extract_arms = Vec::new();
    for (var, _ty) in &ops {
        if var != "CustomPyCrossoverOperatorWrapper" {
            let wrapper = format_ident!("Py{}", var);
            extract_arms.push(quote! {
                if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                    return Ok(#enum_ident::from(extracted.inner));
                }
            });
        }
    }
    // …and only if none of those matched, try the custom wrapper itself
    extract_arms.push(quote! {
        if let Ok(extracted) = py_obj.extract::<CustomPyCrossoverOperatorWrapper>(py) {
            return Ok(#enum_ident::from(extracted));
        }
    });
    let ctor_impl = quote! {
        impl #enum_ident {
            /// Convert a Python-side operator instance into this dispatcher.
            pub fn from_python_operator(
                py_obj: pyo3::PyObject
            ) -> pyo3::PyResult<Self> {
                pyo3::Python::with_gil(|py| {
                    #(#extract_arms)*
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "Could not extract a valid crossover operator",
                    ))
                })
            }
        }
    };

    // Emit: original enum + all generated glue
    TokenStream::from(quote! {
        #input_enum               // keep user enum unchanged
        #(#from_impls)*
        #crossover_impl
        #(#macro_calls)*
        #ctor_impl
    })
}

/// ----------------------------------------------------------------------
///         Registration Macro for Sampling Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Applies to an enum whose variants are of the form `Variant(Type)`. For each
/// variant this attribute will:
/// - Generate `impl From<Type> for SamplingOperatorDispatcher`
/// - Implement `moors::operators::SamplingOperator` by delegating
///   `sample_individual(num_vars, rng)`
/// - Emit a call to `py_operator_sampling!(Type)` so that the Python wrapper is registered
/// - Add an associated constructor:
///     `fn from_python_operator(py_obj: PyObject) -> PyResult<Self>`
///   which extracts the correct variant from a `PyObject`.
#[proc_macro_attribute]
pub fn register_py_operators_sampling(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the enum the user wrote.
    let input_enum: ItemEnum = parse_macro_input!(item as ItemEnum);
    let enum_ident = &input_enum.ident;

    // Collect (VariantIdent, FieldType) for each tuple-variant `Variant(Type)`
    let ops: Vec<(proc_macro2::Ident, Type)> = input_enum
        .variants
        .iter()
        .filter_map(|v| match &v.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => {
                Some((v.ident.clone(), f.unnamed[0].ty.clone()))
            }
            _ => None, // skip unit or struct-like variants
        })
        .collect();

    // impl From<Type> for the enum
    let from_impls = ops.iter().map(|(var, ty)| {
        quote! {
            impl From<#ty> for #enum_ident {
                fn from(op: #ty) -> Self {
                    #enum_ident::#var(op)
                }
            }
        }
    });

    // impl SamplingOperator by delegating sample_individual(...)
    let sample_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.sample_individual(num_vars, rng),
        }
    });
    let operate_match = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.operate(population_size, num_vars, rng),
        }
    });

    let sampling_impl = quote! {
        impl moors::operators::SamplingOperator for #enum_ident {
            fn sample_individual(
                &self,
                num_vars: usize,
                rng: &mut impl moors::random::RandomGenerator
            ) -> ndarray::Array1<f64> {
                match self { #(#sample_match)* }
            }
            fn operate(
                &self,
                population_size: usize,
                num_vars: usize,
                rng: &mut impl moors::random::RandomGenerator
            ) -> ndarray::Array2<f64> {
                match self { #(#operate_match)* }
            }
        }
    };

    // invoke py_operator_sampling!(Type) for each Rust operator type
    let macro_calls = ops.iter().filter_map(|(var, ty)| {
        if var == "CustomPySamplingOperatorWrapper" {
            None
        } else {
            Some(quote! { pymoors_macros::py_operator_sampling!(#ty); })
        }
    });
    // from_python_operator constructor: try the PySampling wrappers first…
    let mut extract_arms = Vec::new();
    for (var, _ty) in &ops {
        if var != "CustomPySamplingOperatorWrapper" {
            let wrapper = format_ident!("Py{}", var);
            extract_arms.push(quote! {
                if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                    return Ok(#enum_ident::from(extracted.inner));
                }
            });
        }
    }
    // …and only if none of those matched, try the custom wrapper itself
    extract_arms.push(quote! {
        if let Ok(extracted) = py_obj.extract::<CustomPySamplingOperatorWrapper>(py) {
            return Ok(#enum_ident::from(extracted));
        }
    });
    let ctor_impl = quote! {
        impl #enum_ident {
            /// Convert a Python-side sampling operator into this dispatcher.
            pub fn from_python_operator(
                py_obj: pyo3::PyObject
            ) -> pyo3::PyResult<Self> {
                pyo3::Python::with_gil(|py| {
                    #(#extract_arms)*
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "Could not extract a valid sampling operator",
                    ))
                })
            }
        }
    };

    // Emit: original enum plus all generated glue
    TokenStream::from(quote! {
        #input_enum
        #(#from_impls)*
        #sampling_impl
        #(#macro_calls)*
        #ctor_impl
    })
}

/// ----------------------------------------------------------------------
///         Registration Macro for Duplicates Operators (Enum Dispatch)
/// ----------------------------------------------------------------------
///
/// Applies to an enum named `DuplicatesCleanerDispatcher` whose variants
/// are tuple-style `Variant(Type)`. This attribute will generate
/// `impl From<Type>` conversions, implement `moors::duplicates::PopulationCleaner`
/// by delegating `remove(...)`, invoke `py_operator_duplicates!(Type)` for each
/// operator type, and add a `from_python_operator(py_obj)` constructor that
/// extracts the correct variant from a Python object.
#[proc_macro_attribute]
pub fn register_py_operators_duplicates(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the user’s enum
    let input_enum: ItemEnum = parse_macro_input!(item as ItemEnum);
    let enum_ident = &input_enum.ident;

    // Gather (variant, type) pairs for each tuple variant
    let ops: Vec<(proc_macro2::Ident, Type)> = input_enum
        .variants
        .iter()
        .filter_map(|v| match &v.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => {
                Some((v.ident.clone(), f.unnamed[0].ty.clone()))
            }
            _ => None,
        })
        .collect();

    // impl From<Type> for the enum
    let from_impls = ops.iter().map(|(var, ty)| {
        quote! {
            impl From<#ty> for #enum_ident {
                fn from(op: #ty) -> Self { #enum_ident::#var(op) }
            }
        }
    });

    // Implement PopulationCleaner by delegating remove(...)
    let remove_arms = ops.iter().map(|(var, _)| {
        quote! {
            #enum_ident::#var(inner) => inner.remove(population, reference),
        }
    });
    let cleaner_impl = quote! {
        impl moors::duplicates::PopulationCleaner for #enum_ident {
            fn remove(
                &self,
                population:ndarray::Array2<f64>,
                reference: Option<&ndarray::Array2<f64>>,
            ) -> ndarray::Array2<f64> {
                match self { #(#remove_arms)* }
            }
        }
    };

    // Emit py_operator_duplicates!(Type) for each operator
    let macro_calls = ops.iter().map(|(_, ty)| {
        quote! { pymoors_macros::py_operator_duplicates!(#ty); }
    });

    // Constructor to extract from PyObject
    let extract_arms = ops.iter().map(|(var, _)| {
        let wrapper = format_ident!("Py{}", var);
        quote! {
            if let Ok(extracted) = py_obj.extract::<#wrapper>(py) {
                return Ok(#enum_ident::from(extracted.inner));
            }
        }
    });
    let ctor_impl = quote! {
        impl #enum_ident {
            /// Convert an optional Python-side duplicates operator into this dispatcher.
            /// If `py_obj_opt` is `None`, returns the `NoDuplicatesCleaner` variant.
            pub fn from_python_operator(
                py_obj_opt: Option<pyo3::PyObject>
            ) -> pyo3::PyResult<Self> {
                // Early return for no-op cleaner
                if py_obj_opt.is_none() {
                    return Ok(
                        #enum_ident::NoDuplicatesCleaner(NoDuplicatesCleaner)
                    );
                }
                let py_obj = py_obj_opt.unwrap();
                pyo3::Python::with_gil(|py| {
                    #(#extract_arms)*
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "Could not extract a valid duplicates operator",
                    ))
                })
            }
        }
    };

    // Emit the original enum plus all generated code
    TokenStream::from(quote! {
        #input_enum
        #(#from_impls)*
        #cleaner_impl
        #(#macro_calls)*
        #ctor_impl
    })
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
                    .map_err(|e| AlgorithmErrorWrapper(e.into()))?;
                Ok(())
            }

            /// Getter for the algorithm's population.
            /// It converts the internal population members (genes, fitness, rank, constraints)
            /// to Python objects using NumPy.
            #[getter]
            pub fn population(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
                let schemas_module = py.import("pymoors.schemas")?;
                let population_class = schemas_module.getattr("Population")?;
                let population = self
                    .algorithm
                    .population()
                    .map_err(|e| AlgorithmErrorWrapper(e.into()))?;
                let py_genes = population.genes.to_pyarray(py);
                let py_fitness = population.fitness.to_pyarray(py);
                let py_constraints = population.constraints.to_pyarray(py);

                let py_rank = if let Some(ref r) = population.rank {
                    r.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };
                let py_survival_score = if let Some(ref r) = population.survival_score {
                    r.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };
                let py_survival_score = if let Some(ref r) = population.survival_score {
                    r.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("genes", py_genes)?;
                kwargs.set_item("fitness", py_fitness)?;
                kwargs.set_item("rank", py_rank)?;
                kwargs.set_item("constraints", py_constraints)?;
                kwargs.set_item("survival_score", py_survival_score)?;
                let py_instance = population_class.call((), Some(&kwargs))?;
                Ok(py_instance.into_py(py))
            }
        }
    };

    TokenStream::from(expanded)
}
