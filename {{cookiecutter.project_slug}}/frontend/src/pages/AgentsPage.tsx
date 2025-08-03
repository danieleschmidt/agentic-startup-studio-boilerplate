import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { agentService } from '@/services/agentService';
import { Brain, Lightbulb, Search, FileText, Loader2 } from 'lucide-react';

interface ResearchRequest {
  task_description: string;
  priority: 'low' | 'normal' | 'high';
  context?: Record<string, any>;
}

/**
 * Agents Page - Interface for AI agent interactions
 */
export default function AgentsPage() {
  const [taskDescription, setTaskDescription] = useState('');
  const [priority, setPriority] = useState<'low' | 'normal' | 'high'>('normal');
  const [context, setContext] = useState('');
  const [asyncMode, setAsyncMode] = useState(false);

  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Mutation for executing research tasks
  const executeResearchMutation = useMutation({
    mutationFn: async (request: ResearchRequest) => {
      if (asyncMode) {
        return await agentService.submitResearchTaskAsync(request);
      } else {
        return await agentService.executeResearchTask(request);
      }
    },
    onSuccess: (data) => {
      if (asyncMode) {
        toast({
          title: 'Task Submitted',
          description: `Task submitted successfully. Task ID: ${data.task_id}`,
        });
      } else {
        toast({
          title: 'Task Completed',
          description: 'Research task executed successfully',
        });
      }
      
      // Reset form
      setTaskDescription('');
      setContext('');
      setPriority('normal');
      
      // Refresh tasks list
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
    onError: (error: any) => {
      toast({
        title: 'Error',
        description: error.message || 'Failed to execute research task',
        variant: 'destructive',
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!taskDescription.trim()) {
      toast({
        title: 'Validation Error',
        description: 'Please provide a task description',
        variant: 'destructive',
      });
      return;
    }

    const request: ResearchRequest = {
      task_description: taskDescription,
      priority,
      context: context ? JSON.parse(context) : undefined,
    };

    executeResearchMutation.mutate(request);
  };

  const agentCapabilities = [
    {
      icon: Search,
      title: 'Research Agent',
      description: 'Gathers comprehensive information and insights on given topics',
      capabilities: ['Web search', 'Data collection', 'Source verification', 'Trend analysis'],
    },
    {
      icon: Brain,
      title: 'Analysis Agent',
      description: 'Processes and analyzes data to extract meaningful patterns',
      capabilities: ['Statistical analysis', 'Pattern recognition', 'Risk assessment', 'Insights generation'],
    },
    {
      icon: FileText,
      title: 'Report Agent',
      description: 'Creates detailed, professional reports based on research and analysis',
      capabilities: ['Report generation', 'Executive summaries', 'Data visualization', 'Recommendations'],
    },
  ];

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI Agents</h1>
          <p className="text-muted-foreground">
            Execute research tasks using our AI agent crew
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="flex items-center space-x-1">
            <Brain className="w-3 h-3" />
            <span>CrewAI Powered</span>
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Task Submission Form */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Lightbulb className="w-5 h-5" />
                <span>Submit Research Task</span>
              </CardTitle>
              <CardDescription>
                Describe the research task you want our AI agents to perform
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="task-description">Task Description</Label>
                  <Textarea
                    id="task-description"
                    placeholder="Describe what you want the agents to research and analyze..."
                    value={taskDescription}
                    onChange={(e) => setTaskDescription(e.target.value)}
                    rows={4}
                    required
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="priority">Priority</Label>
                    <Select value={priority} onValueChange={(value: 'low' | 'normal' | 'high') => setPriority(value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select priority" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="execution-mode">Execution Mode</Label>
                    <Select value={asyncMode ? 'async' : 'sync'} onValueChange={(value) => setAsyncMode(value === 'async')}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sync">Synchronous</SelectItem>
                        <SelectItem value="async">Asynchronous</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="context">Additional Context (Optional)</Label>
                  <Textarea
                    id="context"
                    placeholder='{"industry": "technology", "timeframe": "last 6 months"}'
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    rows={3}
                  />
                  <p className="text-sm text-muted-foreground">
                    Provide additional context as JSON format
                  </p>
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  disabled={executeResearchMutation.isPending}
                >
                  {executeResearchMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {asyncMode ? 'Submitting Task...' : 'Executing Task...'}
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      {asyncMode ? 'Submit Task' : 'Execute Task'}
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Task Result Display */}
          {executeResearchMutation.isSuccess && !asyncMode && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Task Result</CardTitle>
                <CardDescription>
                  Research task completed successfully
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {executeResearchMutation.data?.execution_time?.toFixed(2)}s
                    </div>
                    <div>
                      <span className="font-medium">Agents Used:</span>{' '}
                      {executeResearchMutation.data?.agents_used?.length}
                    </div>
                  </div>
                  <Separator />
                  <div className="prose prose-sm max-w-none">
                    <h4>Research Result:</h4>
                    <pre className="whitespace-pre-wrap bg-muted p-4 rounded-lg text-sm">
                      {executeResearchMutation.data?.result}
                    </pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Agent Capabilities */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Agent Capabilities</h2>
          {agentCapabilities.map((agent, index) => (
            <Card key={index}>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2 text-lg">
                  <agent.icon className="w-5 h-5" />
                  <span>{agent.title}</span>
                </CardTitle>
                <CardDescription className="text-sm">
                  {agent.description}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  {agent.capabilities.map((capability, capIndex) => (
                    <Badge key={capIndex} variant="secondary" className="text-xs">
                      {capability}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}

          {/* Quick Tips */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">💡 Quick Tips</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>• Be specific in your task descriptions for better results</p>
              <p>• Use high priority for urgent research tasks</p>
              <p>• Async mode is recommended for complex, time-consuming tasks</p>
              <p>• Add context JSON to provide domain-specific information</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}