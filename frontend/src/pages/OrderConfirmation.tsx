import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Order } from '@/types/product';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CheckCircle, Package, Clock, MapPin } from 'lucide-react';

const OrderConfirmation = () => {
  const { orderId } = useParams();
  const navigate = useNavigate();
  const [order, setOrder] = useState<Order | null>(null);

  useEffect(() => {
    const orders = JSON.parse(localStorage.getItem('orders') || '[]');
    const foundOrder = orders.find((o: Order) => o.id === orderId);
    if (foundOrder) {
      setOrder(foundOrder);
    } else {
      navigate('/');
    }
  }, [orderId, navigate]);

  if (!order) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-3xl">
        <div className="text-center mb-8 animate-fade-in">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10 mb-4">
            <CheckCircle className="h-8 w-8 text-accent" />
          </div>
          <h1 className="text-3xl font-bold mb-2">Order Confirmed!</h1>
          <p className="text-muted-foreground">
            Thank you for your purchase. We'll send you a confirmation email shortly.
          </p>
        </div>

        {/* Order Details */}
        <Card className="p-6 mb-6 animate-fade-in">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-lg font-semibold mb-1">Order #{order.id}</h2>
              <p className="text-sm text-muted-foreground">
                {new Date(order.createdAt).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground mb-1">Total</p>
              <p className="text-2xl font-bold text-accent">${order.total.toFixed(2)}</p>
            </div>
          </div>

          {/* Delivery Info */}
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-muted/50">
              <Clock className="h-5 w-5 text-accent mt-0.5" />
              <div>
                <p className="font-medium mb-1">Estimated Delivery</p>
                <p className="text-sm text-muted-foreground">{order.estimatedDelivery}</p>
              </div>
            </div>
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-muted/50">
              <Package className="h-5 w-5 text-accent mt-0.5" />
              <div>
                <p className="font-medium mb-1">Order Status</p>
                <p className="text-sm text-muted-foreground capitalize">{order.status}</p>
              </div>
            </div>
          </div>

          {/* Shipping Address */}
          <div className="p-4 rounded-lg bg-muted/50 mb-6">
            <div className="flex items-start space-x-3">
              <MapPin className="h-5 w-5 text-accent mt-0.5" />
              <div>
                <p className="font-medium mb-2">Shipping Address</p>
                <p className="text-sm text-muted-foreground">
                  {order.customerDetails.name}<br />
                  {order.customerDetails.address}<br />
                  {order.customerDetails.city}, {order.customerDetails.zipCode}<br />
                  {order.customerDetails.phone}
                </p>
              </div>
            </div>
          </div>

          {/* Order Items */}
          <div>
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
          </div>
        </Card>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3 animate-fade-in">
          <Button
            onClick={() => navigate(`/track/${order.id}`)}
            className="flex-1 bg-accent hover:bg-accent/90"
            size="lg"
          >
            Track Order
          </Button>
          <Button
            onClick={() => navigate('/')}
            variant="outline"
            className="flex-1"
            size="lg"
          >
            Continue Shopping
          </Button>
        </div>
      </div>
    </div>
  );
};

export default OrderConfirmation;
