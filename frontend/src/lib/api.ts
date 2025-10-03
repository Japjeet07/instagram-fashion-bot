import axios from 'axios';
import { Product } from '@/types/product';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchProducts = async (imageFile: File): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await api.post('/search', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

export const precomputeEmbeddings = async (): Promise<void> => {
  try {
    await api.post('/precompute');
  } catch (error) {
    console.error('Precompute error:', error);
    throw error;
  }
};

// Load inventory from backend
export const getAllProducts = async (): Promise<Product[]> => {
  try {
    const response = await api.get('/inventory', {
      params: { t: Date.now() } // Cache busting
    });
    return response.data;
  } catch (error) {
    console.error('Failed to load inventory:', error);
    // Fallback: return empty array
    return [];
  }
};

export default api;
