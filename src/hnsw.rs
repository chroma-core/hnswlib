use std::{
    ffi::{c_char, c_int, CString},
    path::PathBuf,
    str::Utf8Error,
};
use thiserror::Error;
use uuid::Uuid;

// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
struct HnswIndexPtrFFI {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[link(name = "bindings", kind = "static")]
extern "C" {
    fn create_index(space_name: *const c_char, dim: c_int) -> *const HnswIndexPtrFFI;

    fn free_index(index: *const HnswIndexPtrFFI);

    fn init_index(
        index: *const HnswIndexPtrFFI,
        max_elements: usize,
        M: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: bool,
        is_persistent: bool,
        path: *const c_char,
    );

    fn load_index(
        index: *const HnswIndexPtrFFI,
        path: *const c_char,
        allow_replace_deleted: bool,
        is_persistent_index: bool,
        max_elements: usize,
    );

    fn persist_dirty(index: *const HnswIndexPtrFFI);

    fn add_item(index: *const HnswIndexPtrFFI, data: *const f32, id: usize, replace_deleted: bool);
    fn mark_deleted(index: *const HnswIndexPtrFFI, id: usize);
    fn get_item(index: *const HnswIndexPtrFFI, id: usize, data: *mut f32);
    fn get_all_ids_sizes(index: *const HnswIndexPtrFFI, sizes: *mut usize);
    fn get_all_ids(
        index: *const HnswIndexPtrFFI,
        non_deleted_ids: *mut usize,
        deleted_ids: *mut usize,
    );
    fn knn_query(
        index: *const HnswIndexPtrFFI,
        query_vector: *const f32,
        k: usize,
        ids: *mut usize,
        distance: *mut f32,
        allowed_ids: *const usize,
        allowed_ids_length: usize,
        disallowed_ids: *const usize,
        disallowed_ids_length: usize,
    ) -> c_int;
    fn open_fd(index: *const HnswIndexPtrFFI);
    fn close_fd(index: *const HnswIndexPtrFFI);
    fn get_ef(index: *const HnswIndexPtrFFI) -> c_int;
    fn set_ef(index: *const HnswIndexPtrFFI, ef: c_int);
    fn len(index: *const HnswIndexPtrFFI) -> c_int;
    fn len_with_deleted(index: *const HnswIndexPtrFFI) -> c_int;
    fn capacity(index: *const HnswIndexPtrFFI) -> c_int;
    fn resize_index(index: *const HnswIndexPtrFFI, new_size: usize);
    fn get_last_error(index: *const HnswIndexPtrFFI) -> *const c_char;
}

#[derive(Error, Debug)]
pub enum HnswError {
    // A generic C++ exception, stores the error message
    #[error("HnswError: `{0}`")]
    FFIException(String),
    #[error(transparent)]
    ErrorStringRead(#[from] Utf8Error),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HnswDistanceFunction {
    Euclidean,
    Cosine,
    InnerProduct,
}

impl From<HnswDistanceFunction> for String {
    fn from(df: HnswDistanceFunction) -> Self {
        match df {
            HnswDistanceFunction::Euclidean => "l2".to_string(),
            HnswDistanceFunction::Cosine => "cosine".to_string(),
            HnswDistanceFunction::InnerProduct => "ip".to_string(),
        }
    }
}

#[derive(Error, Debug)]
pub enum HnswInitError {
    #[error("Invalid distance function `{0}`")]
    InvalidDistanceFunction(String),
    #[error("Invalid path `{0}`. Are you sure the path exists?")]
    InvalidPath(String),
    #[error(transparent)]
    Other(#[from] HnswError),
}

pub struct HnswIndex {
    id: Uuid,
    ffi_ptr: *const HnswIndexPtrFFI,
    dimensionality: i32,
}

pub enum HnswCreationMode {
    Initialize,
    Load,
}

pub const DEFAULT_MAX_ELEMENTS: usize = 10000;

pub struct HnswIndexConfig {
    pub id: Uuid,
    pub distance_function: HnswDistanceFunction,
    pub dimensionality: i32,
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub random_seed: usize,
    pub persist_path: Option<PathBuf>,
    pub mode: HnswCreationMode,
}

impl HnswIndex {
    pub fn try_new(config: HnswIndexConfig) -> Result<Self, HnswInitError> {
        let distance_function_string: String = config.distance_function.into();
        let space_name = CString::new(distance_function_string)
            .map_err(|e| HnswInitError::InvalidDistanceFunction(e.to_string()))?;

        let ffi_ptr = unsafe { create_index(space_name.as_ptr(), config.dimensionality) };
        read_and_return_hnsw_error(ffi_ptr)?;

        let path = match config.persist_path.clone() {
            Some(path) => path
                .to_str()
                .ok_or_else(|| HnswInitError::InvalidPath("Invalid UTF-8 path".to_string()))?
                .to_string(),
            None => "".to_string(),
        };

        let path = CString::new(path).map_err(|e| HnswInitError::InvalidPath(e.to_string()))?;

        match config.mode {
            HnswCreationMode::Initialize => {
                unsafe {
                    init_index(
                        ffi_ptr,
                        config.max_elements,
                        config.m,
                        config.ef_construction,
                        config.random_seed,
                        true,
                        config.persist_path.is_some(),
                        path.as_ptr(),
                    );
                }
                read_and_return_hnsw_error(ffi_ptr)?;
            }
            HnswCreationMode::Load => {
                unsafe {
                    load_index(ffi_ptr, path.as_ptr(), true, true, DEFAULT_MAX_ELEMENTS);
                }
                read_and_return_hnsw_error(ffi_ptr)?;
            }
        }

        let hnsw_index = HnswIndex {
            ffi_ptr,
            dimensionality: config.dimensionality,
            id: config.id,
        };
        hnsw_index.set_ef(config.ef_search)?;
        Ok(hnsw_index)
    }

    pub fn len(&self) -> usize {
        unsafe { len(self.ffi_ptr) as usize }
        // Does not return an error
    }

    pub fn len_with_deleted(&self) -> usize {
        unsafe { len_with_deleted(self.ffi_ptr) as usize }
        // Does not return an error
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dimensionality(&self) -> i32 {
        self.dimensionality
    }

    pub fn capacity(&self) -> usize {
        unsafe { capacity(self.ffi_ptr) as usize }
        // Does not return an error
    }

    pub fn resize(&mut self, new_size: usize) -> Result<(), HnswError> {
        unsafe { resize_index(self.ffi_ptr, new_size) }
        read_and_return_hnsw_error(self.ffi_ptr)
    }

    pub fn open_fd(&self) {
        unsafe { open_fd(self.ffi_ptr) }
    }

    pub fn close_fd(&self) {
        unsafe { close_fd(self.ffi_ptr) }
    }

    pub fn get_all_ids_sizes(&self) -> Result<Vec<usize>, HnswError> {
        let mut sizes = vec![0usize; 2];
        unsafe { get_all_ids_sizes(self.ffi_ptr, sizes.as_mut_ptr()) };
        read_and_return_hnsw_error(self.ffi_ptr)?;
        Ok(sizes)
    }

    pub fn get_all_ids(&self) -> Result<(Vec<usize>, Vec<usize>), HnswError> {
        let sizes = self.get_all_ids_sizes()?;
        let mut non_deleted_ids = vec![0usize; sizes[0]];
        let mut deleted_ids = vec![0usize; sizes[1]];
        unsafe {
            get_all_ids(
                self.ffi_ptr,
                non_deleted_ids.as_mut_ptr(),
                deleted_ids.as_mut_ptr(),
            );
        }
        read_and_return_hnsw_error(self.ffi_ptr)?;
        Ok((non_deleted_ids, deleted_ids))
    }

    pub fn add(&self, id: usize, vector: &[f32]) -> Result<(), HnswError> {
        unsafe { add_item(self.ffi_ptr, vector.as_ptr(), id, true) }
        read_and_return_hnsw_error(self.ffi_ptr)
    }

    pub fn delete(&self, id: usize) -> Result<(), HnswError> {
        unsafe { mark_deleted(self.ffi_ptr, id) }
        read_and_return_hnsw_error(self.ffi_ptr)
    }

    pub fn get(&self, id: usize) -> Result<Option<Vec<f32>>, HnswError> {
        unsafe {
            let mut data: Vec<f32> = vec![0.0f32; self.dimensionality as usize];
            get_item(self.ffi_ptr, id, data.as_mut_ptr());
            read_and_return_hnsw_error(self.ffi_ptr)?;
            Ok(Some(data))
        }
    }

    pub fn query(
        &self,
        vector: &[f32],
        k: usize,
        allowed_ids: &[usize],
        disallowed_ids: &[usize],
    ) -> Result<(Vec<usize>, Vec<f32>), HnswError> {
        let actual_k = std::cmp::min(k, self.len());
        let mut ids = vec![0usize; actual_k];
        let mut distance = vec![0.0f32; actual_k];
        let total_result = unsafe {
            knn_query(
                self.ffi_ptr,
                vector.as_ptr(),
                k,
                ids.as_mut_ptr(),
                distance.as_mut_ptr(),
                allowed_ids.as_ptr(),
                allowed_ids.len(),
                disallowed_ids.as_ptr(),
                disallowed_ids.len(),
            ) as usize
        };
        read_and_return_hnsw_error(self.ffi_ptr)?;

        if total_result < actual_k {
            ids.truncate(total_result);
            distance.truncate(total_result);
        }
        Ok((ids, distance))
    }

    pub fn save(&self) -> Result<(), HnswError> {
        unsafe { persist_dirty(self.ffi_ptr) };
        read_and_return_hnsw_error(self.ffi_ptr)?;
        Ok(())
    }

    // fn get_ef(&self) -> Result<usize, HnswError> {
    //     let ret_val;
    //     unsafe { ret_val = get_ef(self.ffi_ptr) as usize }
    //     read_and_return_hnsw_error(self.ffi_ptr)?;
    //     Ok(ret_val)
    // }

    fn set_ef(&self, ef: usize) -> Result<(), HnswError> {
        unsafe { set_ef(self.ffi_ptr, ef as c_int) }
        read_and_return_hnsw_error(self.ffi_ptr)
    }
}

fn read_and_return_hnsw_error(ffi_ptr: *const HnswIndexPtrFFI) -> Result<(), HnswError> {
    let err = unsafe { get_last_error(ffi_ptr) };
    if !err.is_null() {
        match unsafe { std::ffi::CStr::from_ptr(err).to_str() } {
            Ok(err_str) => return Err(HnswError::FFIException(err_str.to_string())),
            Err(e) => return Err(HnswError::ErrorStringRead(e)),
        }
    }
    Ok(())
}

impl Drop for HnswIndex {
    fn drop(&mut self) {
        unsafe { free_index(self.ffi_ptr) }
    }
}
