/**
 * Agent Service
 * Handles API communication for AI agent operations
 */

import { apiClient } from './apiClient';

export interface AgentRequest {
  task_description: string;
  priority: 'low' | 'normal' | 'high';
  context?: Record<string, any>;
  deadline?: string;
}

export interface AgentResponse {
  success: boolean;
  result: string;
  metadata: Record<string, any>;
  execution_time: number;
  agents_used: string[];
}

export interface TaskSubmissionResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  progress: number;
  result?: string;
  error?: string;
}

/**
 * Agent service for AI operations
 */
export const agentService = {
  /**
   * Execute a research task synchronously
   */
  async executeResearchTask(request: AgentRequest): Promise<AgentResponse> {
    const response = await apiClient.post<AgentResponse>('/agents/research', request);
    return response.data;
  },

  /**
   * Submit a research task for asynchronous execution
   */
  async submitResearchTaskAsync(request: AgentRequest): Promise<TaskSubmissionResponse> {
    const response = await apiClient.post<TaskSubmissionResponse>('/agents/research/async', request);
    return response.data;
  },

  /**
   * Get the status of a specific task
   */
  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await apiClient.get<TaskStatusResponse>(`/agents/tasks/${taskId}`);
    return response.data;
  },

  /**
   * List all tasks for the current user
   */
  async listUserTasks(): Promise<TaskStatusResponse[]> {
    const response = await apiClient.get<TaskStatusResponse[]>('/agents/tasks');
    return response.data;
  },

  /**
   * Cancel or delete a task
   */
  async cancelTask(taskId: string): Promise<{ message: string }> {
    const response = await apiClient.delete<{ message: string }>(`/agents/tasks/${taskId}`);
    return response.data;
  },

  /**
   * Validate agent request data
   */
  validateRequest(request: AgentRequest): string[] {
    const errors: string[] = [];

    if (!request.task_description?.trim()) {
      errors.push('Task description is required');
    }

    if (request.task_description && request.task_description.length < 10) {
      errors.push('Task description must be at least 10 characters long');
    }

    if (request.task_description && request.task_description.length > 5000) {
      errors.push('Task description must be less than 5000 characters');
    }

    if (!['low', 'normal', 'high'].includes(request.priority)) {
      errors.push('Priority must be low, normal, or high');
    }

    if (request.deadline) {
      const deadlineDate = new Date(request.deadline);
      const now = new Date();
      if (deadlineDate <= now) {
        errors.push('Deadline must be in the future');
      }
    }

    return errors;
  },

  /**
   * Format task result for display
   */
  formatTaskResult(result: string): string {
    try {
      // Try to parse as JSON for better formatting
      const parsed = JSON.parse(result);
      return JSON.stringify(parsed, null, 2);
    } catch {
      // Return as-is if not JSON
      return result;
    }
  },

  /**
   * Get priority color for UI display
   */
  getPriorityColor(priority: string): string {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'normal':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'low':
        return 'text-gray-600 bg-gray-50 border-gray-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  },

  /**
   * Get status color for UI display
   */
  getStatusColor(status: string): string {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'running':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'pending':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  },

  /**
   * Calculate estimated completion time based on task complexity
   */
  estimateCompletionTime(taskDescription: string, priority: string): number {
    const baseTime = 30; // Base time in seconds
    const wordCount = taskDescription.split(' ').length;
    
    // Complexity multiplier based on word count
    let complexityMultiplier = 1;
    if (wordCount > 100) {
      complexityMultiplier = 2;
    } else if (wordCount > 50) {
      complexityMultiplier = 1.5;
    }

    // Priority multiplier
    let priorityMultiplier = 1;
    if (priority === 'high') {
      priorityMultiplier = 0.8; // High priority tasks are processed faster
    } else if (priority === 'low') {
      priorityMultiplier = 1.2; // Low priority tasks take longer
    }

    return Math.round(baseTime * complexityMultiplier * priorityMultiplier);
  },

  /**
   * Parse context string into object
   */
  parseContext(contextString: string): Record<string, any> | null {
    if (!contextString?.trim()) {
      return null;
    }

    try {
      return JSON.parse(contextString);
    } catch (error) {
      throw new Error('Invalid JSON format in context field');
    }
  },

  /**
   * Get agent recommendations based on task description
   */
  getAgentRecommendations(taskDescription: string): string[] {
    const recommendations: string[] = [];
    const lowerDesc = taskDescription.toLowerCase();

    if (lowerDesc.includes('research') || lowerDesc.includes('investigate') || lowerDesc.includes('find')) {
      recommendations.push('Use specific keywords for better search results');
    }

    if (lowerDesc.includes('analyze') || lowerDesc.includes('compare') || lowerDesc.includes('evaluate')) {
      recommendations.push('Provide data sources or criteria for analysis');
    }

    if (lowerDesc.includes('report') || lowerDesc.includes('summary') || lowerDesc.includes('document')) {
      recommendations.push('Specify the target audience for the report');
    }

    if (lowerDesc.includes('trend') || lowerDesc.includes('forecast') || lowerDesc.includes('predict')) {
      recommendations.push('Include time range for trend analysis');
    }

    if (recommendations.length === 0) {
      recommendations.push('Be specific about desired outputs and format');
    }

    return recommendations;
  }
};