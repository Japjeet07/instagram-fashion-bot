import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Order } from '@/types/product';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Package, CheckCircle, Truck, Home, Search } from 'lucide-react';

const OrderTracking = () => {
  const { orderId: paramOrderId } = useParams();
  const navigate = useNavigate();
  const [orderId, setOrderId] = useState(paramOrderId || '');
  const [order, setOrder] = useState<Order | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (paramOrderId) {
      searchOrder(paramOrderId);
    }
  }, [paramOrderId]);

  const searchOrder = (id: string) => {
    const orders = JSON.parse(localStorage.getItem('orders') || '[]');
    const foundOrder = orders.find((o: Order) => o.id === id);
    if (foundOrder) {
      setOrder(foundOrder);
      setNotFound(false);
    } else {
      setOrder(null);
      setNotFound(true);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (orderId.trim()) {
      searchOrder(orderId.trim());
    }
  };

  const getStatusSteps = (status: Order['status']) => {
    const steps = [
      { key: 'pending', label: 'Order Placed', icon: Package },
      { key: 'processing', label: 'Processing', icon: CheckCircle },
      { key: 'shipped', label: 'Shipped', icon: Truck },
      { key: 'delivered', label: 'Delivered', icon: Home },
    ];

    const statusIndex = steps.findIndex((s) => s.key === status);
    return steps.map((step, index) => ({
      ...step,
      completed: index <= statusIndex,
      active: index === statusIndex,
    }));
  };

  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-3xl">
        <h1 className="text-3xl font-bold mb-8 animate-fade-in">Track Your Order</h1>

        {/* Search Form */}
        <Card className="p-6 mb-8 animate-fade-in">
          <form onSubmit={handleSearch}>
            <Label htmlFor="orderId" className="mb-2 block">
              Order ID
            </Label>
            <div className="flex gap-2">
              <Input
                id="orderId"
                value={orderId}
                onChange={(e) => setOrderId(e.target.value)}
                placeholder="Enter your order ID (e.g., ORD-1234567890)"
                className="flex-1"
              />
              <Button type="submit" className="bg-accent hover:bg-accent/90">
                <Search className="h-4 w-4 mr-2" />
                Track
              </Button>
            </div>
          </form>
        </Card>

        {/* Not Found Message */}
        {notFound && !order && (
          <Card className="p-8 text-center animate-fade-in">
            <Package className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">Order Not Found</h2>
            <p className="text-muted-foreground mb-6">
              We couldn't find an order with that ID. Please check and try again.
            </p>
            <Button onClick={() => navigate('/')} variant="outline">
              Go to Shop
            </Button>
          </Card>
        )}

        {/* Order Tracking */}
        {order && (
          <div className="space-y-6 animate-fade-in">
            {/* Order Info */}
            <Card className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-xl font-semibold mb-1">Order #{order.id}</h2>
                  <p className="text-sm text-muted-foreground">
                    Placed on{' '}
                    {new Date(order.createdAt).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    })}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-muted-foreground mb-1">Estimated Delivery</p>
                  <p className="font-semibold">{order.estimatedDelivery}</p>
                </div>
              </div>

              {/* Progress Steps */}
              <div className="relative">
                <div className="absolute top-5 left-0 right-0 h-0.5 bg-border">
                  <div
                    className="h-full bg-accent transition-all duration-500"
                    style={{
                      width: `${
                        (getStatusSteps(order.status).filter((s) => s.completed).length /
                          getStatusSteps(order.status).length) *
                        100
                      }%`,
                    }}
                  />
                </div>

                <div className="relative grid grid-cols-4 gap-4">
                  {getStatusSteps(order.status).map((step) => {
                    const Icon = step.icon;
                    return (
                      <div key={step.key} className="flex flex-col items-center">
                        <div
                          className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 transition-colors ${
                            step.completed
                              ? 'bg-accent text-accent-foreground'
                              : 'bg-muted text-muted-foreground'
                          }`}
                        >
                          <Icon className="h-5 w-5" />
                        </div>
                        <p
                          className={`text-xs text-center font-medium ${
                            step.completed ? 'text-foreground' : 'text-muted-foreground'
                          }`}
                        >
                          {step.label}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </Card>

            {/* Order Items */}
            <Card className="p-6">
              <h3 className="font-semibold mb-4">Order Items</h3>
              <div className="space-y-4">
                {order.items.map((item) => (
                  <div key={item.id} className="flex gap-4">
                    <img
                      src={item.image_path}
                      alt={item.name}
                      className="h-20 w-20 rounded object-cover"
                    />
                    <div className="flex-1">
                      <p className="font-medium">{item.name}</p>
                      <p className="text-sm text-muted-foreground">
                        Quantity: {item.quantity}
                      </p>
                      <p className="text-sm font-semibold mt-1">
                        ${(item.price * item.quantity).toFixed(2)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Delivery Address */}
            <Card className="p-6">
              <h3 className="font-semibold mb-4">Delivery Address</h3>
              <p className="text-sm text-muted-foreground">
                {order.customerDetails.name}<br />
                {order.customerDetails.address}<br />
                {order.customerDetails.city}, {order.customerDetails.zipCode}<br />
                {order.customerDetails.phone}
              </p>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default OrderTracking;
