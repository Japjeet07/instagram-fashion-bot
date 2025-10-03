export interface Product {
  id: string;
  name: string;
  price: number;
  image_path: string;
  url?: string;
  delivery: "30min" | "2-3 days";
  category: string;
  similarity_score?: number;
}

export interface CartItem extends Product {
  quantity: number;
}

export interface Order {
  id: string;
  items: CartItem[];
  total: number;
  customerDetails: CustomerDetails;
  status: "pending" | "processing" | "shipped" | "delivered";
  createdAt: string;
  estimatedDelivery: string;
}

export interface CustomerDetails {
  name: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  zipCode: string;
}
