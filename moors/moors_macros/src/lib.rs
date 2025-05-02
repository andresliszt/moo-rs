//! Generates <AlgorithmName>Builder from an `impl … { pub fn new(..) -> Result<Self, E> }`.
//!
//! Literal defaults inside build():
//!   • keep_infeasible = false
//!   • verbose         = false
//!   • crossover_rate  = 0.9
//!   • mutation_rate   = 0.1

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, parse_quote, FnArg, GenericArgument, ImplItem, ImplItemFn,
    ItemImpl, Pat, PathArguments, ReturnType, Type,
};

#[proc_macro_attribute]
pub fn algorithm_builder(_attr: TokenStream, item: TokenStream) -> TokenStream {
    /* ── parse the `impl … { fn new… }` block ────────────────────────── */
    let impl_block: ItemImpl = parse_macro_input!(item as ItemImpl);

    // extract the algorithm name, e.g. `Nsga2`
    let algo_ident = match &*impl_block.self_ty {
        Type::Path(p) => &p.path.segments.last().unwrap().ident,
        _ => panic!("algorithm_builder: expected concrete type"),
    };
    // builder will be `Nsga2Builder`
    let builder_ident: Ident = format_ident!("{algo_ident}Builder", span = Span::call_site());

    /* ── original generics / where-clause ─────────────────────────────── */
    let generics = impl_block.generics.clone();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    /* ── find the `fn new` constructor ───────────────────────────────── */
    let new_fn: ImplItemFn = impl_block
        .items
        .iter()
        .find_map(|it| match it {
            ImplItem::Fn(f) if f.sig.ident == "new" => Some(f.clone()),
            _ => None,
        })
        .expect("algorithm_builder: `fn new` not found");

    /* ── collect constructor params into builder fields ──────────── */
    struct Field { ident: Ident, ty_inner: Type, is_optional: bool }
    let mut fields = Vec::new();
    for arg in &new_fn.sig.inputs {
        if let FnArg::Typed(pat_ty) = arg {
            let ident = match &*pat_ty.pat {
                Pat::Ident(id) => id.ident.clone(),
                _ => panic!("algorithm_builder: complex patterns not supported"),
            };
            let ty = *pat_ty.ty.clone();
            let (inner_ty, is_optional) = match &ty {
                Type::Path(tp)
                    if tp.path.segments.len() == 1 && tp.path.segments[0].ident == "Option" =>
                {
                    if let PathArguments::AngleBracketed(ab) = &tp.path.segments[0].arguments {
                        let first = ab.args.first().expect("Option without generic");
                        match first {
                            GenericArgument::Type(t) => (t.clone(), true),
                            _ => panic!("algorithm_builder: unexpected generic arg"),
                        }
                    } else {
                        (ty.clone(), false)
                    }
                }
                _ => (ty.clone(), false),
            };
            fields.push(Field { ident, ty_inner: inner_ty, is_optional });
        }
    }

    /* ── extract the error type `E` from `Result<Self, E>` ─────────────── */
    let err_ty: Type = match &new_fn.sig.output {
        ReturnType::Type(_, boxed) => match &**boxed {
            Type::Path(tp) if tp.path.segments.last().unwrap().ident == "Result" => {
                if let PathArguments::AngleBracketed(ab) = &tp.path.segments.last().unwrap().arguments {
                    match &ab.args[1] {
                        GenericArgument::Type(t) => t.clone(),
                        _ => parse_quote!(()),
                    }
                } else { parse_quote!(()) }
            }
            _ => parse_quote!(()),
        },
        ReturnType::Default => parse_quote!(()),
    };

    /* ── builder struct fields & default values ───────────────────────── */
    let builder_fields = fields.iter().map(|f| {
        let name = &f.ident;
        let ty   = &f.ty_inner;
        quote!( #name: ::std::option::Option<#ty> )
    });
    let builder_defaults = fields.iter().map(|f| {
        let name = &f.ident;
        if f.is_optional {
            let ty = &f.ty_inner;
            quote!( #name: ::std::option::Option::<#ty>::None )
        } else {
            quote!( #name: ::std::option::Option::None )
        }
    });

    /* ── fluent setters ───────────────────────────────────────────────── */
    let setter_methods = fields.iter().map(|f| {
        let ident = &f.ident;
        let ty    = &f.ty_inner;
        quote! {
            pub fn #ident(mut self, value: #ty) -> Self {
                self.#ident = ::std::option::Option::Some(value);
                self
            }
        }
    });

    /* ── literal defaults for non-Option params ───────────────────────── */
    fn default_expr(field: &str) -> Option<proc_macro2::TokenStream> {
        match field {
            "keep_infeasible" | "verbose" => Some(quote!(false)),
            "crossover_rate"              => Some(quote!(0.9)),
            "mutation_rate"               => Some(quote!(0.1)),
            _ => None,
        }
    }

    /* ── build() arguments ────────────────────────────────────────────── */
    let build_args = fields.iter().map(|f| {
        let name = &f.ident;
        let key  = name.to_string();
        if f.is_optional {
            quote!( self.#name )
        } else if let Some(def) = default_expr(&key) {
            quote!( self.#name.unwrap_or(#def) )
        } else {
            let msg = format!("{} is required", key);
            quote!( self.#name.expect(#msg) )
        }
    });

    /* ── final builder implementation ────────────────────────────────── */
    let builder_code = quote! {
        #[derive(Debug)]
        pub struct #builder_ident #impl_generics #where_clause {
            #( #builder_fields ),*
        }

        impl #impl_generics ::std::default::Default for #builder_ident #ty_generics #where_clause {
            fn default() -> Self {
                Self { #( #builder_defaults ),* }
            }
        }

        impl #impl_generics #builder_ident #ty_generics #where_clause {
            #( #setter_methods )*

            pub fn build(self) -> ::std::result::Result<
                #algo_ident #ty_generics,
                #err_ty
            > {
                #algo_ident::#ty_generics::new( #( #build_args ),* )
            }
        }
    };

    /* ── emit original impl + generated builder ───────────────────────── */
    let expanded = quote! {
        #impl_block
        #builder_code
    };
    TokenStream::from(expanded)
}
