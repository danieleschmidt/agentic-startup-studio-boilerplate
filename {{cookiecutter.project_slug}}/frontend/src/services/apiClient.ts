/**
 * API Client Configuration
 * Centralized HTTP client for backend communication
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add request timestamp for debugging
    config.metadata = { startTime: new Date() };
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Add response time for debugging
    const endTime = new Date();
    const startTime = response.config.metadata?.startTime;
    if (startTime) {
      console.log(`API ${response.config.method?.toUpperCase()} ${response.config.url}: ${endTime.getTime() - startTime.getTime()}ms`);
    }
    
    return response;
  },
  (error) => {
    // Handle common HTTP errors
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem('auth_token');
          if (window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
          break;
          
        case 403:
          // Forbidden
          console.error('Access forbidden:', data.detail || 'Insufficient permissions');
          break;
          
        case 404:
          // Not found
          console.error('Resource not found:', error.config.url);
          break;
          
        case 422:
          // Validation error
          console.error('Validation error:', data.detail || data);
          break;
          
        case 429:
          // Rate limited
          console.error('Rate limit exceeded. Please try again later.');
          break;
          
        case 500:
          // Server error
          console.error('Server error:', data.detail || 'Internal server error');
          break;
          
        default:
          console.error('API error:', status, data);
      }
      
      // Transform error for consistent handling
      const apiError = {
        message: data.detail || data.message || 'An error occurred',
        status,
        data,
      };
      
      return Promise.reject(apiError);
    } else if (error.request) {
      // Network error
      console.error('Network error:', error.message);
      return Promise.reject({
        message: 'Network error. Please check your connection.',
        status: 0,
        data: null,
      });
    } else {
      // Other error
      console.error('Request error:', error.message);
      return Promise.reject({
        message: error.message || 'An unexpected error occurred',
        status: 0,
        data: null,
      });
    }
  }
);

// Enhanced API client with typed methods
class ApiClient {
  private client: AxiosInstance;

  constructor(axiosInstance: AxiosInstance) {
    this.client = axiosInstance;
  }

  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.get<T>(url, config);
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.post<T>(url, data, config);
  }

  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.put<T>(url, data, config);
  }

  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.patch<T>(url, data, config);
  }

  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.delete<T>(url, config);
  }

  // Health check method
  async healthCheck(): Promise<{ status: string; service: string }> {
    const response = await this.get('/health');
    return response.data;
  }

  // Set authentication token
  setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token);
  }

  // Clear authentication token
  clearAuthToken(): void {
    localStorage.removeItem('auth_token');
  }

  // Get current auth token
  getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  // Check if user is authenticated
  isAuthenticated(): boolean {
    return !!this.getAuthToken();
  }
}

// Export the enhanced API client
export const apiClient = new ApiClient(apiClient);
export default apiClient;