import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, Bot, CheckSquare, Clock, AlertCircle, TrendingUp } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { apiClient } from '@/services/apiClient';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

interface UserStats {
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  active_api_keys: number;
}

export default function DashboardPage() {
  const { user } = useAuth();

  // Fetch user statistics
  const { data: stats, isLoading: statsLoading } = useQuery<UserStats>({
    queryKey: ['userStats'],
    queryFn: async () => {
      const response = await apiClient.get('/users/me/stats');
      return response.data;
    },
  });

  // Fetch recent tasks
  const { data: recentTasks, isLoading: tasksLoading } = useQuery({
    queryKey: ['recentTasks'],
    queryFn: async () => {
      const response = await apiClient.get('/agents/tasks?limit=5');
      return response.data;
    },
  });

  const getCompletionRate = () => {
    if (!stats || stats.total_tasks === 0) return 0;
    return Math.round((stats.completed_tasks / stats.total_tasks) * 100);
  };

  const getSuccessRate = () => {
    if (!stats || stats.total_tasks === 0) return 0;
    const successfulTasks = stats.completed_tasks;
    return Math.round((successfulTasks / stats.total_tasks) * 100);
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Welcome back, {user?.full_name || user?.username}!
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Here's an overview of your AI agent activities
          </p>
        </div>
        <div className="flex space-x-4">
          <Button asChild>
            <Link to="/agents">
              <Bot className="w-4 h-4 mr-2" />
              Run Agent
            </Link>
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tasks</CardTitle>
            <CheckSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statsLoading ? '...' : stats?.total_tasks || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              All-time task executions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {statsLoading ? '...' : stats?.completed_tasks || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {getCompletionRate()}% completion rate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed</CardTitle>
            <AlertCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {statsLoading ? '...' : stats?.failed_tasks || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Tasks that encountered errors
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <BarChart3 className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {statsLoading ? '...' : `${getSuccessRate()}%`}
            </div>
            <p className="text-xs text-muted-foreground">
              Overall success rate
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>
              Common tasks you can perform
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button asChild className="w-full justify-start">
              <Link to="/agents">
                <Bot className="w-4 h-4 mr-2" />
                Create New Agent Task
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/tasks">
                <CheckSquare className="w-4 h-4 mr-2" />
                View All Tasks
              </Link>
            </Button>
            <Button asChild variant="outline" className="w-full justify-start">
              <Link to="/settings">
                <BarChart3 className="w-4 h-4 mr-2" />
                Manage Settings
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>
              Your latest agent task executions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {tasksLoading ? (
              <div className="space-y-2">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="animate-pulse">
                    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                    <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                  </div>
                ))}
              </div>
            ) : recentTasks && recentTasks.length > 0 ? (
              <div className="space-y-4">
                {recentTasks.slice(0, 5).map((task: any) => (
                  <div key={task.id} className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      task.status === 'completed' ? 'bg-green-500' :
                      task.status === 'failed' ? 'bg-red-500' :
                      task.status === 'running' ? 'bg-blue-500' :
                      'bg-gray-400'
                    }`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {task.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {task.agent_type} • {new Date(task.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {task.status}
                    </div>
                  </div>
                ))}
                <Button asChild variant="outline" className="w-full mt-4">
                  <Link to="/tasks">View All Tasks</Link>
                </Button>
              </div>
            ) : (
              <div className="text-center py-8">
                <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500 dark:text-gray-400">
                  No tasks yet. Create your first agent task!
                </p>
                <Button asChild className="mt-4">
                  <Link to="/agents">Get Started</Link>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Performance Chart Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Overview</CardTitle>
          <CardDescription>
            Task execution trends over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 dark:text-gray-400">
                Performance charts coming soon
              </p>
              <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">
                Track your agent performance over time
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}