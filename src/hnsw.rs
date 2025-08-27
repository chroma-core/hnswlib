use std::{
    ffi::{c_char, c_int, c_uchar, CString},
    path::PathBuf,
    str::Utf8Error,
};
use thiserror::Error;

// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
struct HnswIndexPtrFFI {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

// Opaque struct for memory buffers
#[repr(C)]
pub struct DataFFI {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

// Read-only version of DataFFI
#[repr(C)]
pub struct DataViewFFI {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*const u8, core::marker::PhantomPinned)>,
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

    
    // Functions for Rust-managed buffer allocation
    fn get_required_header_size(index: *const HnswIndexPtrFFI) -> usize;
    fn get_required_data_level0_size(index: *const HnswIndexPtrFFI) -> usize;
    fn get_required_length_size(index: *const HnswIndexPtrFFI) -> usize;
    fn get_required_link_list_size(index: *const HnswIndexPtrFFI) -> usize;

    fn create_mutable_hnsw_data(header_buffer: *mut c_uchar, header_size: usize, data_level0_buffer: *mut c_uchar, data_level0_size: usize, length_buffer: *mut c_uchar, length_size: usize, link_list_buffer: *mut c_uchar, link_list_size: usize) -> *const DataFFI;
    fn create_hnsw_data_view(header_buffer: *const c_uchar, header_size: usize, data_level0_buffer: *const c_uchar, data_level0_size: usize, length_buffer: *const c_uchar, length_size: usize, link_list_buffer: *const c_uchar, link_list_size: usize) -> *const DataViewFFI;
    fn hnsw_data_to_view(hnsw_data :*const DataFFI) -> *const DataViewFFI;

    fn serialize_index_to_hnsw_data(index: *const HnswIndexPtrFFI, hnsw_data: *const DataFFI) -> bool;
    fn load_index_from_hnsw_data(index: *const HnswIndexPtrFFI, buffers: *const DataViewFFI, max_elements: usize);
}

#[derive(Error, Debug)]
pub enum HnswError {
    // A generic C++ exception, stores the error message
    #[error("HnswError: `{0}`")]
    FFIException(String),
    #[error(transparent)]
    ErrorStringRead(#[from] Utf8Error),
    #[error("HnswDataError: `{0}`")]
    ErrorHnswData(String)
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
    ffi_ptr: *const HnswIndexPtrFFI,
    dimensionality: i32,
}

// Make index sync, we should wrap index so that it is sync in the way we expect but for now this implements the trait
unsafe impl Sync for HnswIndex {}
unsafe impl Send for HnswIndex {}

pub struct HnswIndexLoadConfig {
    pub distance_function: HnswDistanceFunction,
    pub dimensionality: i32,
    pub persist_path: PathBuf,
    pub ef_search: usize,
    pub max_elements: usize,
}

pub struct HnswIndexInitConfig {
    pub distance_function: HnswDistanceFunction,
    pub dimensionality: i32,
    pub persist_path: Option<PathBuf>,
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub random_seed: usize,
}

impl HnswIndex {
    pub fn init(config: HnswIndexInitConfig) -> Result<Self, HnswInitError> {
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

        let hnsw_index = HnswIndex {
            ffi_ptr,
            dimensionality: config.dimensionality,
        };
        hnsw_index.set_ef(config.ef_search)?;
        Ok(hnsw_index)
    }

    pub fn load(config: HnswIndexLoadConfig) -> Result<Self, HnswInitError> {
        let distance_function_string: String = config.distance_function.into();
        let space_name = CString::new(distance_function_string)
            .map_err(|e| HnswInitError::InvalidDistanceFunction(e.to_string()))?;

        let ffi_ptr = unsafe { create_index(space_name.as_ptr(), config.dimensionality) };
        read_and_return_hnsw_error(ffi_ptr)?;

        let path = config
            .persist_path
            .clone()
            .to_str()
            .ok_or_else(|| HnswInitError::InvalidPath("Invalid UTF-8 path".to_string()))?
            .to_string();

        let path = CString::new(path).map_err(|e| HnswInitError::InvalidPath(e.to_string()))?;

        unsafe {
            load_index(ffi_ptr, path.as_ptr(), true, true, config.max_elements);
        }
        read_and_return_hnsw_error(ffi_ptr)?;

        let hnsw_index = HnswIndex {
            ffi_ptr,
            dimensionality: config.dimensionality,
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

    pub fn get_ef(&self) -> Result<usize, HnswError> {
        let ret_val;
        unsafe { ret_val = get_ef(self.ffi_ptr) as usize }
        read_and_return_hnsw_error(self.ffi_ptr)?;
        Ok(ret_val)
    }

    fn set_ef(&self, ef: usize) -> Result<(), HnswError> {
        unsafe { set_ef(self.ffi_ptr, ef as c_int) }
        read_and_return_hnsw_error(self.ffi_ptr)
    }

    /// Serialize the index to memory buffers with Rust-managed allocation
    pub fn serialize_index_to_hnsw_data(&self) -> Result<HnswData, HnswError> {
        HnswData::new_from_index(self.ffi_ptr)
    }

    /// Load index from memory buffers
    pub fn load_from_hnsw_data(config: HnswIndexLoadConfig, load_data: HnswData) -> Result<Self, HnswInitError> {
        let distance_function_string: String = config.distance_function.into();
        let space_name = CString::new(distance_function_string)
            .map_err(|e| HnswInitError::InvalidDistanceFunction(e.to_string()))?;

        let ffi_ptr = unsafe { create_index(space_name.as_ptr(), config.dimensionality) };
        read_and_return_hnsw_error(ffi_ptr)?;

        unsafe {
            load_index_from_hnsw_data(ffi_ptr, load_data.ffi_ptr, config.max_elements);
        }
        read_and_return_hnsw_error(ffi_ptr)?;

        let hnsw_index = HnswIndex {
            ffi_ptr,
            dimensionality: config.dimensionality,
        };
        hnsw_index.set_ef(config.ef_search)?;
        Ok(hnsw_index)
    }
}

/// Safe wrapper for memory buffers containing serialized HNSW index data
pub struct HnswData {
    ffi_ptr: *const DataViewFFI,
    _marker: std::marker::PhantomData<*mut ()>, // Prevents Copy trait

    // Hold Arc references to prevent premature buffer deallocation
    header_buffer: std::sync::Arc<Vec<u8>>,
    data_level0_buffer: std::sync::Arc<Vec<u8>>,
    length_buffer: std::sync::Arc<Vec<u8>>,
    link_list_buffer: std::sync::Arc<Vec<u8>>,
}

unsafe impl Sync for HnswData {}
unsafe impl Send for HnswData {}

impl HnswData {
    /// Create new HnswData with Rust-managed buffers from index serialization
    fn new_from_index(
        index_ptr: *const HnswIndexPtrFFI,
    ) -> Result<Self, HnswError> {
        // Allocate mutable buffers in Rust
        let header_size = unsafe { get_required_header_size(index_ptr) };
        let data_level0_size = unsafe { get_required_data_level0_size(index_ptr) };
        let length_size = unsafe { get_required_length_size(index_ptr) };
        let link_list_size = unsafe { get_required_link_list_size(index_ptr) };
        
        let mut header_buffer = vec![0u8; header_size];
        let mut data_level0_buffer = vec![0u8; data_level0_size];
        let mut length_buffer = vec![0u8; length_size];
        let mut link_list_buffer = vec![0u8; link_list_size];

        let ffi_ptr = unsafe { create_mutable_hnsw_data(header_buffer.as_mut_ptr(), header_size, data_level0_buffer.as_mut_ptr(), data_level0_size, length_buffer.as_mut_ptr(), length_size, link_list_buffer.as_mut_ptr(), link_list_size) };
        
        read_and_return_hnsw_error(index_ptr)?;
        
        // Call C++ to serialize directly into our Rust-allocated buffers
        let success = unsafe {
            serialize_index_to_hnsw_data(index_ptr, ffi_ptr)
        };
        
        if !success {
            return Err(HnswError::FFIException("Failed to serialize to external buffers".to_string()));
        }
        
        Ok(HnswData {
            ffi_ptr: unsafe { hnsw_data_to_view(ffi_ptr) },
            _marker: std::marker::PhantomData,
            header_buffer: std::sync::Arc::new(header_buffer),
            data_level0_buffer: std::sync::Arc::new(data_level0_buffer),
            length_buffer: std::sync::Arc::new(length_buffer),
            link_list_buffer: std::sync::Arc::new(link_list_buffer),
        })
    }

    /// Create new HnswData from existing Arc buffer pointers
    fn new_from_buffers(
        header_buffer: std::sync::Arc<Vec<u8>>,
        data_level0_buffer: std::sync::Arc<Vec<u8>>,
        length_buffer: std::sync::Arc<Vec<u8>>,
        link_list_buffer: std::sync::Arc<Vec<u8>>,
    ) -> Result<Self, HnswError> {
        // Create HnswData FFI structure and set the buffers
        let ffi_ptr = unsafe { create_hnsw_data_view(header_buffer.as_ptr(), header_buffer.len(), data_level0_buffer.as_ptr(), data_level0_buffer.len(), length_buffer.as_ptr(), length_buffer.len(), link_list_buffer.as_ptr(), link_list_buffer.len()) };
        if ffi_ptr.is_null() {
            return Err(HnswError::ErrorHnswData("Failed to create HnswData structure".to_string()));
        }
        
        Ok(HnswData {
            ffi_ptr,
            _marker: std::marker::PhantomData,
            header_buffer: header_buffer,
            data_level0_buffer: data_level0_buffer,
            length_buffer: length_buffer,
            link_list_buffer: link_list_buffer,
        })
    }

    /// Get the header buffer as a byte slice
    pub fn header_buffer(&self) -> &[u8] {
        &self.header_buffer
    }

    /// Get the data level 0 buffer as a byte slice
    pub fn data_level0_buffer(&self) -> &[u8] {
        &self.data_level0_buffer
    }

    /// Get the length buffer as a byte slice
    pub fn length_buffer(&self) -> &[u8] {
        &self.length_buffer
    }

    /// Get the link list buffer as a byte slice
    pub fn link_list_buffer(&self) -> &[u8] {
        &self.link_list_buffer
    }

    /// Create a new HnswDataBuilder for safe construction
    pub fn builder() -> HnswDataBuilder {
        HnswDataBuilder::new()
    }
}

/// Builder for safely constructing HnswData instances
pub struct HnswDataBuilder {
    header_buffer: Option<std::sync::Arc<Vec<u8>>>,
    data_level0_buffer: Option<std::sync::Arc<Vec<u8>>>,
    length_buffer: Option<std::sync::Arc<Vec<u8>>>,
    link_list_buffer: Option<std::sync::Arc<Vec<u8>>>,
}

impl HnswDataBuilder {
    pub fn new() -> Self {
        Self {
            header_buffer: None,
            data_level0_buffer: None,
            length_buffer: None,
            link_list_buffer: None,
        }
    }

    pub fn header_buffer(mut self, buffer: std::sync::Arc<Vec<u8>>) -> Self {
        self.header_buffer = Some(buffer);
        self
    }

    pub fn data_level0_buffer(mut self, buffer: std::sync::Arc<Vec<u8>>) -> Self {
        self.data_level0_buffer = Some(buffer);
        self
    }
    
    pub fn length_buffer(mut self, buffer: std::sync::Arc<Vec<u8>>) -> Self {
        self.length_buffer = Some(buffer);
        self
    }

    pub fn link_list_buffer(mut self, buffer: std::sync::Arc<Vec<u8>>) -> Self {
        self.link_list_buffer = Some(buffer);
        self
    }

    /// Validate that all required buffers are set and non-empty
    fn validate(&self) -> Result<(), HnswError> {
        if self.header_buffer.is_none() {
            return Err(HnswError::ErrorHnswData("Header buffer is required".to_string()));
        }
        if self.data_level0_buffer.is_none() {
            return Err(HnswError::ErrorHnswData("Data level0 buffer is required".to_string()));
        }
        if self.length_buffer.is_none() {
            return Err(HnswError::ErrorHnswData("Length buffer is required".to_string()));
        }
        if self.link_list_buffer.is_none() {
            return Err(HnswError::ErrorHnswData("Link list buffer is required".to_string()));
        }

        // Check that buffers are not empty
        if self.header_buffer.as_ref().unwrap().is_empty() {
            return Err(HnswError::ErrorHnswData("Header buffer cannot be empty".to_string()));
        }
        if self.data_level0_buffer.as_ref().unwrap().is_empty() {
            return Err(HnswError::ErrorHnswData("Data level0 buffer cannot be empty".to_string()));
        }
        if self.length_buffer.as_ref().unwrap().is_empty() {
            return Err(HnswError::ErrorHnswData("Length buffer cannot be empty".to_string()));
        }

        // It's ok for link list buffer to be empty in an empty index.

        Ok(())
    }

    /// Build the HnswData instance after validating all buffers are set
    pub fn build(self) -> Result<HnswData, HnswError> {
        self.validate()?;

        HnswData::new_from_buffers(
            self.header_buffer.unwrap(),
            self.data_level0_buffer.unwrap(),
            self.length_buffer.unwrap(),
            self.link_list_buffer.unwrap(),
        )
    }
}

impl Default for HnswDataBuilder {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
pub mod test {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;

    use super::*;
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use tempfile::tempdir;

    const EPS: f32 = 0.00001;

    fn generate_random_data(n: usize, d: usize) -> Vec<f32> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let mut data = vec![0.0f32; n * d];
        // Generate random data
        for i in 0..n {
            for j in 0..d {
                data[i * d + j] = rng.gen();
            }
        }
        data
    }

    fn index_data_same(index: &HnswIndex, ids: &[usize], data: &[f32], dim: usize) {
        for (i, id) in ids.iter().enumerate() {
            let actual_data = index.get(*id);
            match actual_data {
                Ok(actual_data) => match actual_data {
                    None => panic!("No data found for id: {}", id),
                    Some(actual_data) => {
                        assert_eq!(actual_data.len(), dim);
                        for j in 0..dim {
                            // Floating point epsilon comparison
                            assert!((actual_data[j] - data[i * dim + j]).abs() < EPS);
                        }
                    }
                },
                Err(_) => panic!("Did not expect error"),
            }
        }
    }

    #[test]
    fn it_initializes_and_can_set_get_ef() {
        let n = 1000;
        let d: usize = 960;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function: HnswDistanceFunction::Euclidean,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 10,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });
        match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => {
                assert_eq!(index.get_ef().unwrap(), 10);
                index.set_ef(100).expect("Should not error");
                assert_eq!(index.get_ef().unwrap(), 100);
            }
        }
    }

    #[test]
    fn it_can_add_parallel() {
        let n: usize = 100;
        let d: usize = 960;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function: HnswDistanceFunction::InnerProduct,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let ids: Vec<usize> = (0..n).collect();

        // Add data in parallel, using global pool for testing
        ThreadPoolBuilder::new()
            .num_threads(12)
            .build_global()
            .unwrap();

        let data = generate_random_data(n, d);

        (0..n).into_par_iter().for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        assert_eq!(index.len(), n);

        // Get the data and check it
        index_data_same(&index, &ids, &data, d);
    }

    #[test]
    fn it_can_add_and_basic_query() {
        let n = 1;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };
        assert_eq!(index.get_ef().unwrap(), 100);

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Assert length
        assert_eq!(index.len(), n);

        // Get the data and check it
        index_data_same(&index, &ids, &data, d);

        // Query the data
        let query = &data[0..d];
        let allow_ids = &[];
        let disallow_ids = &[];
        let (ids, distances) = index.query(query, 1, allow_ids, disallow_ids).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(ids[0], 0);
        assert_eq!(distances[0], 0.0);
    }

    #[test]
    fn it_can_add_and_delete() {
        let n = 1000;
        let d = 960;

        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        assert_eq!(index.len(), n);

        // Delete some of the data
        let mut rng = rand::thread_rng();
        let delete_ids: Vec<usize> = (0..n).choose_multiple(&mut rng, n / 20);

        for id in &delete_ids {
            index.delete(*id).expect("Should not error");
        }

        assert_eq!(index.len(), n - delete_ids.len());

        let allow_ids = &[];
        let disallow_ids = &[];
        // Query for the deleted ids and ensure they are not found
        for deleted_id in &delete_ids {
            let target_vector = &data[*deleted_id * d..(*deleted_id + 1) * d];
            let (ids, _) = index
                .query(target_vector, 10, allow_ids, disallow_ids)
                .unwrap();
            for check_deleted_id in &delete_ids {
                assert!(!ids.contains(check_deleted_id));
            }
        }
    }

    #[test]
    fn it_can_persist_and_load() {
        let n = 1000;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Persist the index
        let res = index.save();
        if let Err(e) = res {
            panic!("Error saving index: {}", e);
        }

        // Load the index
        let index = HnswIndex::load(HnswIndexLoadConfig {
            distance_function,
            dimensionality: d as i32,
            persist_path: persist_path.to_path_buf(),
            ef_search: 100,
            max_elements: 100,
        });

        let index = match index {
            Err(e) => panic!("Error loading index: {}", e),
            Ok(index) => index,
        };
        assert_eq!(index.get_ef().expect("Expected to get ef_search"), 100);

        // Query the data
        let query = &data[0..d];
        let allow_ids = &[];
        let disallow_ids = &[];
        let (ids, distances) = index.query(query, 1, allow_ids, disallow_ids).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(ids[0], 0);
        assert_eq!(distances[0], 0.0);

        // Get the data and check it
        index_data_same(&index, &ids, &data, d);
    }

    #[test]
    fn it_can_persist_and_load_from_memory() {
        let n = 1000;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Persist the index
        let res = index.save();
        if let Err(e) = res {
            panic!("Error saving index: {}", e);
        }

        // Load the index from memory instead
        let files = ["header", "data_level0", "length", "link_lists"];
        let ext = "bin";
        let mut src_buffers = Vec::new();

        for file in files {
            let path = persist_path.join(file).with_extension(ext);
            let data = std::fs::read(path).expect("Unable to read file");
            src_buffers.push(Arc::new(data));
        }

        let hnsw_data = HnswDataBuilder::new()
            .header_buffer(src_buffers[0].clone())
            .data_level0_buffer(src_buffers[1].clone())
            .length_buffer(src_buffers[2].clone())
            .link_list_buffer(src_buffers[3].clone())
            .build();

        let index = HnswIndex::load_from_hnsw_data(HnswIndexLoadConfig {
            distance_function,
            dimensionality: d as i32,
            persist_path: "".into(),
            ef_search: 100,
            max_elements: 100,
        }, hnsw_data.expect("Failed to create HnswData"));

        let index = match index {
            Err(e) => panic!("Error loading index: {}", e),
            Ok(index) => index,
        };
        assert_eq!(index.get_ef().expect("Expected to get ef_search"), 100);

        // Query the data
        let query = &data[0..d];
        let allow_ids = &[];
        let disallow_ids = &[];
        let (ids, distances) = index.query(query, 1, allow_ids, disallow_ids).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(ids[0], 0);
        assert_eq!(distances[0], 0.0);

        // Get the data and check it
        index_data_same(&index, &ids, &data, d);
    }

    #[test]
    fn it_can_serialize_and_deserialize_hnsw_data() {
        let n = 100;
        let d: usize = 128;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        
        // Create and populate original index
        let original_index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 42,
            persist_path: Some(persist_path.to_path_buf()),
        }).expect("Failed to create original index");

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        // Add data to original index
        for i in 0..n {
            let data_slice = &data[i * d..(i + 1) * d];
            original_index.add(ids[i], data_slice).expect("Should not error");
        }

        // Verify original index has correct data
        assert_eq!(original_index.len(), n);
        index_data_same(&original_index, &ids, &data, d);

        // Serialize to memory buffers
        let hnsw_data = original_index.serialize_index_to_hnsw_data()
            .expect("Failed to serialize to memory buffers");

        // Verify buffers are not empty
        assert!(!hnsw_data.header_buffer().is_empty(), "Header buffer should not be empty");
        assert!(!hnsw_data.data_level0_buffer().is_empty(), "Data level0 buffer should not be empty");
        assert!(!hnsw_data.length_buffer().is_empty(), "Length buffer should not be empty");
        assert!(!hnsw_data.link_list_buffer().is_empty(), "Link list buffer should not be empty");

        // Create new index from memory buffers
        let loaded_index = HnswIndex::load_from_hnsw_data(
            HnswIndexLoadConfig {
                distance_function,
                dimensionality: d as i32,
                persist_path: "".into(),
                ef_search: 100,
                max_elements: 100,
            },
            hnsw_data,
        ).expect("Failed to load from memory buffers");

        // Verify loaded index has same data
        assert_eq!(loaded_index.len(), n);
        index_data_same(&loaded_index, &ids, &data, d);

        // Test querying both indices to ensure they behave the same
        let query_vector = &data[0..d]; // Use first vector as query
        let k = 5;
        
        let (original_ids, original_distances) = original_index.query(query_vector, k, &[], &[])
            .expect("Query should not error");
        let (loaded_ids, loaded_distances) = loaded_index.query(query_vector, k, &[], &[])
            .expect("Query should not error");

        // Results should be identical
        assert_eq!(original_ids, loaded_ids, "Query results should be identical");
        assert_eq!(original_distances.len(), loaded_distances.len());
        for (orig_dist, loaded_dist) in original_distances.iter().zip(loaded_distances.iter()) {
            assert!((orig_dist - loaded_dist).abs() < EPS, "Distances should be nearly identical");
        }
    }

    #[test]
    fn it_can_add_and_query_with_allowed_and_disallowed_ids() {
        let n = 1000;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Query the data
        let query = &data[0..d];
        let allow_ids = &[0, 2];
        let disallow_ids = &[3];
        let (ids, distances) = index.query(query, 10, allow_ids, disallow_ids).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn it_can_resize() {
        let n = 1000;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let mut index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(2 * n, d);
        let ids: Vec<usize> = (0..2 * n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });
        assert_eq!(index.capacity(), n);

        // Resize the index to 2*n
        index.resize(2 * n).expect("Should not error");

        assert_eq!(index.len(), n);
        assert_eq!(index.capacity(), 2 * n);

        // Add another n elements from n to 2n
        (n..2 * n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });
    }

    #[test]
    fn it_can_catch_error() {
        let n = 10;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Add more elements than the index can hold
        let data = &data[0..d];
        let res = index.add(n, data);
        match res {
            Err(_) => {}
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    // TODO(rescrv,sicheng):  This test should be re-enabled once we have a way to detect
    // corruption.
    #[ignore]
    fn it_can_detect_corruption() {
        let n = 1000;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        // Persist the index
        let res = index.save();
        if let Err(e) = res {
            panic!("Error saving index: {}", e);
        }

        // Corrupt the linked list
        let link_list_path = persist_path.join("link_lists.bin");
        let mut link_list_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(link_list_path)
            .unwrap();
        link_list_file.write_all(&u32::MAX.to_le_bytes()).unwrap();

        // Load the corrupted index
        let index = HnswIndex::load(HnswIndexLoadConfig {
            distance_function,
            dimensionality: d as i32,
            persist_path: persist_path.to_path_buf(),
            ef_search: 100,
            max_elements: 100,
        });

        assert!(index.is_err());
        assert!(index
            .map(|_| ())
            .unwrap_err()
            .to_string()
            .contains("HNSW Integrity failure"))
    }

    #[test]
    fn it_can_resize_correctly() {
        let n: usize = 10;
        let d: usize = 960;
        let distance_function = HnswDistanceFunction::Euclidean;
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path();
        let index = HnswIndex::init(HnswIndexInitConfig {
            distance_function,
            dimensionality: d as i32,
            max_elements: n,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            random_seed: 0,
            persist_path: Some(persist_path.to_path_buf()),
        });

        let mut index = match index {
            Err(e) => panic!("Error initializing index: {}", e),
            Ok(index) => index,
        };

        let data: Vec<f32> = generate_random_data(n, d);
        let ids: Vec<usize> = (0..n).collect();

        (0..n).for_each(|i| {
            let data = &data[i * d..(i + 1) * d];
            index.add(ids[i], data).expect("Should not error");
        });

        index.delete(0).unwrap();
        let data = &data[d..2 * d];

        let index_len = index.len_with_deleted();
        let index_capacity = index.capacity();
        if index_len + 1 > index_capacity {
            index.resize(index_capacity * 2).unwrap();
        }
        // this will fail if the index is not resized correctly
        index.add(100, data).unwrap();
    }
}
